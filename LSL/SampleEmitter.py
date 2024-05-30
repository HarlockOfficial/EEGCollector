import hashlib
import random
import sys
import time
import uuid

import numpy as np
import torch
from moabb.datasets import PhysionetMI
from pylsl import StreamInfo, StreamOutlet

import DatasetAugmentation.utils
import VirtualController

# Note: using (both_)hands as resting state
x, _, stream_channel_count, _, _ = DatasetAugmentation.utils.load_dataset(PhysionetMI, events=['hands'], get_verbose_information=True)

path_to_generators = sys.argv[1]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
right_generator, left_generator, feet_generator = VirtualController.load_generators(path_to_generators, device=device)

stream_name = "SampleEEG"
stream_type = "EEG"
stream_sample_rate = DatasetAugmentation.utils.SAMPLE_RATE
stream_channel_format = 'float32'
stream_device_id = uuid.UUID(hashlib.md5('DatasetSampleSimulatedDeviceUUID1'.encode('utf-8')).hexdigest())
info = StreamInfo(stream_name, stream_type, stream_channel_count, stream_sample_rate, stream_channel_format, stream_device_id)

outlet = StreamOutlet(info)

def _on_key_down(generator, key):
    samples = int(0.5 * stream_sample_rate + 1)
    while key == VirtualController.KeyStatus.DOWN:
        seed = torch.rand([1, 1, stream_channel_count, samples]).to(device).to(torch.float32)
        data = generator(seed)
        data = data.detach().cpu().numpy()
        data = np.squeeze(data, axis=1)
        assert data.shape == (1, stream_channel_count, samples)
        outlet.push_chunk(data)
        time.sleep(samples / stream_sample_rate)

def _on_key_up(key):
    samples = int(0.5 * stream_sample_rate + 1)
    while key == VirtualController.KeyStatus.UP:
        sample = random.choice(x)
        sample = np.expand_dims(sample, axis=0)
        assert sample.shape == (1, stream_channel_count, samples)
        outlet.push_chunk(sample)
        time.sleep(samples / stream_sample_rate)

VirtualController.setup_keyboard_input(right_generator, left_generator, feet_generator, None, None, device, on_key_down=_on_key_down, on_key_up=_on_key_up)
