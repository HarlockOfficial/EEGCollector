import pathlib
import pickle
import sys
import time
import datetime
from typing import Generator

import numpy as np

import DatasetAugmentation.utils
import EEGClassificator.utils
import EEGCollector.eego_sdk.utils
import VirtualController
from EEGCollector.eego_sdk import eeg_mapping
from EEGCollector.eego_sdk.eego_sdk_pybind11 import eego_sdk

t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
save_path = pathlib.Path(f'live_data/{t}/')
del t

def amplifier_to_id(amplifier):
  return '{}-{:06d}-{}'.format(amplifier.getType(), amplifier.getFirmwareVersion(), amplifier.getSerialNumber())
def get_amplifier(select_amplifier=None):
    try:
        factory = eego_sdk.factory()

        v = factory.getVersion()
        print('version: {}.{}.{}.{}'.format(v.major, v.minor, v.micro, v.build))

        print('delaying to allow slow devices to attach...')
        time.sleep(1)

        amplifiers = factory.getAmplifiers()
        print('found {} amplifiers'.format(len(amplifiers)))

        assert len(amplifiers) > 0, "No amplifiers found"

        if select_amplifier:
            amplifier = select_amplifier(amplifiers)
        else:
            amplifier = amplifiers[0]
        del amplifiers
        del factory
        return amplifier
    except AssertionError as e:
        print('amplifier error: {}'.format(e))
        del amplifiers
        del factory
        exit(1)
def get_stream(amplifier):
    try:
        rates = amplifier.getSamplingRatesAvailable()
        ref_ranges = amplifier.getReferenceRangesAvailable()
        # note: bipolar range must be 2.5 times the value of the reference range
        bip_ranges = amplifier.getBipolarRangesAvailable()
        print('amplifier: {}'.format(amplifier_to_id(amplifier)))
        print('  rates....... {}'.format(rates))
        print('  ref ranges.. {}'.format(ref_ranges))
        print('  bip ranges.. {}'.format(bip_ranges))
        print('  channels.... {}'.format(amplifier.getChannelList()))

        ps = amplifier.getPowerState()
        print('  power:')
        print('    is powered...... {}'.format(ps.is_powered))
        print('    is charging..... {}'.format(ps.is_charging))
        print('    charging level.. {}'.format(ps.charging_level))

        # select right rate, ideally multiple of 128
        # rate = rates[0]
        # NOTE: takes the highest rate that is a multiple of 128
        rate = list(sorted([r for r in rates if r % 128 == 0], reverse=True))[0]
        del rates
        del ref_ranges
        del bip_ranges
        del ps
        assert rate % 128 == 0, "rate must be multiple of 128"

        stream = amplifier.OpenEegStream(rate)
        print('stream:')
        print('  rate:       {}'.format(rate))
        print('  channels:   {}'.format(stream.getChannelList()))

        return stream, rate
    except AssertionError as e:
        print('stream error: {}'.format(e))
        exit(2)

def get_data(stream, rate) -> Generator[np.ndarray, None, None]:
    try:
        t0 = time.time()
        interval = 1.0 / rate
        tnext = t0
        while True:
            tnext = tnext + interval
            delay = tnext - time.time()
            if delay > 0:
                time.sleep(delay)

            # note: data in this call are returned in volts
            data = stream.getData()
            # also, data are in the format:
            # sample 0: chan 0, chan 1, chan 2, ... , chan n
            # sample 1: chan 0, chan 1, chan 2, ... , chan n
            # sample 2: chan 0, chan 1, chan 2, ... , chan n
            # therefore, rotate the matrix!
            sample_count = data.getSampleCount()
            channel_count = data.getChannelCount()
            print('[{:04.4f}] delay={:03} buffer, channels: {:03} samples: {:03}'
                  .format(time.time() - t0, delay, channel_count, sample_count))

            array = np.empty((channel_count, sample_count))
            assert np.all(array == None), "array has values"
            for s in range(sample_count):
                for c in range(channel_count):
                    eeg_sample = data.getSample(c, s)
                    array[c, s] = eeg_sample.value
            assert np.all(array != None), "array has None values"
            with open(save_path / f'raw_data_{tnext}.pkl', 'wb') as f:
                pickle.dump(array, f)
            array = DatasetAugmentation.utils.to_mV(array)
            array = DatasetAugmentation.utils.data_filter(array, rate=rate)
            with open(save_path / f'initially_processed_data_{tnext}.pkl', 'wb') as f:
                pickle.dump(array, f)
            yield array
    except AssertionError as e:
        print('stream error: {}'.format(e))
        return None
    except KeyboardInterrupt:
        print('stream interrupted')
        raise KeyboardInterrupt
    except Exception as e:
        print('stream error: {}'.format(e))
        return None


def delete_old_data(data:np.ndarray, rate:int, seconds:int):
    return data[:, int(-rate*seconds):]

def select_sample(data:np.ndarray, rate:int, seconds:float):
    return data[:, int(-rate*seconds):]

def down_sample(data:np.ndarray, rate:int, new_rate:int):
    assert new_rate < rate, "new rate must be less than the original rate"
    assert rate % new_rate == 0, "new rate must be a factor of the original rate"
    return data[:, ::int(rate/new_rate)]


def filter_channels(sample: np.ndarray, current_channel_list, channel_list:list[str]= None, mapping=None):
    if channel_list is None:
        channel_list = DatasetAugmentation.utils.ALL_EEG_CHANNELS
    if mapping is None:
        mapping = eeg_mapping
    assert sample.shape[0] == len(current_channel_list), "channel count mismatch"
    # select only the channels that are in the channel list
    channel_list = EEGCollector.eego_sdk.utils.channel_names_to_indices(channel_list, mapping.eeg_to_electrode, mapping.electrode_to_channel)
    return sample[channel_list]

def get_sample() -> Generator[np.ndarray, None, None]:
    amplifier = get_amplifier(lambda amplifiers: amplifiers[0])
    stream, rate = get_stream(amplifier)
    print(f'amplifier: {amplifier_to_id(amplifier)}, stream: {stream}, rate: {rate} Hz, channels: {stream.getChannelList()}')
    try:
        all_data_buffer = np.empty((0, stream.getChannelCount()))
        while data := get_data(stream, rate):
            if data is None:
                break
            all_data_buffer = np.hstack((all_data_buffer, data))
            all_data_buffer = delete_old_data(all_data_buffer, rate, 2)
            sample = select_sample(all_data_buffer, rate, 0.5)
            sample = down_sample(sample, rate, 128)
            sample = filter_channels(sample, stream.getChannelList())
            with open(save_path / f'final_data_{time.time()}.pkl', 'wb') as f:
                pickle.dump(sample, f)
            yield sample
    except AssertionError as e:
        print('stream error: {}'.format(e))
    except KeyboardInterrupt:
        del rate
        del stream
        del amplifier
        raise KeyboardInterrupt
    except Exception as e:
        print('stream error: {}'.format(e))
    finally:
        del rate
        del stream
        del amplifier
        return None

def main(path_to_classificator, url_to_websocket_server):
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    stream_channel_count = DatasetAugmentation.utils.INPUT_CHANNELS
    stream_sample_rate = DatasetAugmentation.utils.SAMPLE_RATE
    samples = int(0.5 * stream_sample_rate + 1)

    classificator = VirtualController.load_classificator(path_to_classificator)
    connection = VirtualController.connect_to_websocket_server(url_to_websocket_server)

    while sample := get_sample():
        if sample is None:
            break
        sample = np.array([sample])
        assert sample.shape == (1, stream_channel_count, samples)
        classification = classificator.predict(sample)[0]
        category = EEGClassificator.utils.from_categorical(classification.item())
        connection.send(category)
        with open(save_path / f'classification_{time.time()}_{category}.pkl', 'wb') as f:
            pickle.dump(classification, f)
        print(f'time:{time.time()}, classification: {category}', flush=True)


if __name__ == '__main__':
    path_to_classificator = sys.argv[1]
    url_to_websocket_server = sys.argv[2]
    main(path_to_classificator, url_to_websocket_server)
