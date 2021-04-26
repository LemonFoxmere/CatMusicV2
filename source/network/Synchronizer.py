import numpy as np

class sync:
    # FOR RAW MIDS ONLY
    @staticmethod
    def trim_front(input):
        trimmed = False
        result = []
        for i in input:
            if(trimmed == True or i[0] != 0):
                result.append(i)
                trimmed = True
        return np.array(result)

    # create a set of training and label data
    # TODO: VERIFY
    @staticmethod
    def sync_data(train, label, chunk_length, chunk_duration=0.125, rate=44000):
        # calculate the amount of wavSamples per chunk
        chunk_size = int(rate * chunk_duration)
        data_set = []
        for i in range(chunk_length-1):
            range_start = i * chunk_size
            range_end = (i+1) * chunk_size
            train_chunk = train[range_start, range_end]
            label_chunk = label[i]
            data_set.append(np.array([label_chunk, train_chunk]))
        return np.array(np.array(data_set))
