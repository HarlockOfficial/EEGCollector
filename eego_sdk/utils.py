def channel_names_to_indices(channel_list: list[str], eeg_to_electrode: dict[str, str], electrode_to_channel: dict[str, int]):
    channel_list = [eeg_to_electrode[channel] for channel in channel_list]
    channel_list = [electrode_to_channel[channel] for channel in channel_list]
    return channel_list