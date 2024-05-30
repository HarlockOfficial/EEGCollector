import sys

from pylsl import StreamInlet, resolve_stream

import DatasetAugmentation.utils
import EEGClassificator.utils
import VirtualController

stream_channel_count = 58
stream_sample_rate = DatasetAugmentation.utils.SAMPLE_RATE
samples = int(0.5 * stream_sample_rate + 1)

path_to_classificator = sys.argv[1]
url_to_websocket_server = sys.argv[2]
classificator = VirtualController.load_classificator(path_to_classificator)
connection = VirtualController.connect_to_websocket_server(url_to_websocket_server)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')

# create a new inlet to read from the stream
buf_len_sec = 0.5
max_samples = 2
inlet = StreamInlet(streams[0], max_buflen=buf_len_sec)

try:
    while True:
        # get a new sample (you can also omit the timestamp part if you're not interested in it)
        sample, _ = inlet.pull_chunk(timeout=max_samples*buf_len_sec, max_samples=max_samples)
        if sample is None:
            continue
        assert sample.shape == (1, stream_channel_count, samples)
        classification = classificator.predict(sample)[0]
        classification = EEGClassificator.utils.from_categorical(classification.item())
        connection.send(classification)
except KeyboardInterrupt:
    connection.close()
    sys.exit(0)
except Exception as e:
    connection.close()
    raise e
