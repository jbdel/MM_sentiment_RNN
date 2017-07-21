import tensorflow as tf
import sys
import random
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import math
_EOS = b"_EOS"




def vector_to_one_hot(vector):
    one_hot = []
    for i in vector:
        if i < 0.0:
            one_hot.append([1,0])
        if i > 0.0:
            one_hot.append([0,1])
    return np.array(one_hot)




def read_data(source_path, target_path):
  data_set = []
  y = []
  src_sum = 0
  src_max = 0
  num_size_discard = 0
  num_zero_discard = 0

  with tf.gfile.GFile(source_path, mode="r") as source_file:
      with tf.gfile.GFile(target_path, mode="r") as target_file:
          source, target = source_file.readline(), target_file.readline()
          counter = 0
          while source and target:
            if counter % 100000 == 0:
              print("  reading data line %d" % counter)
              sys.stdout.flush()
            source_ids = [int(x) for x in source.split()]
            # print(source_ids)
            if (len(source_ids) > 65):
                print("Not taking line ",counter,"(0-index based) of size ",len(source_ids))
                num_size_discard +=1
            elif float(target) == 0.0:
                print("Not taking line ", counter, "(0-index based), because its neutral opinion")
                num_zero_discard +=1
            elif counter == 1919:
                print("Not taking line ", counter, "(0-index based), because its VCslbP0mgZI_1")
            else:
                src_sum += len(source_ids)
                if (len(source_ids) > src_max):
                    src_max = len(source_ids)
                data_set.append(source_ids)
                y.append(float(target))
            counter += 1
            source, target = source_file.readline(), target_file.readline()


  assert len(data_set) == len(y) ,("X and Y dont have same size lel")
  print("num_size_discard", num_size_discard)
  print("num_zero_discard", num_zero_discard)

  print("Lines read : ", counter)
  print("Lines taken : ", len(data_set))
  print("Average tokens per src_sentence : ", src_sum / counter)
  print("Max src_sentence length : ", src_max)
  print("Saving filtered labels y")
  with open(target_path.replace("target", "target_filtered"), "w") as f_f:
      for y_f in y:
          f_f.write(str(y_f) + "\n")



  return np.array(data_set), np.array(y), src_max



#return random (and therefore shuffled) batch from x
def get_batch_random(x,y,batch_size,max_time, PAD_ID):
    assert len(x) == len(y), ("X and Y dont have same size lel")

    x_batch, y_batch, x_length = [], [], []
    for i in range(batch_size):
        r = random.randrange(0, len(x), 2)
        encoder_pad = [PAD_ID] * (max_time - len(x[r]))

        if len(x[r]) != max_time:
            if isinstance(PAD_ID,np.ndarray):
                example = (np.concatenate([x[r],encoder_pad]))
            else:
                example = ([x[r] + encoder_pad])
        else:
            if isinstance(PAD_ID, np.ndarray):
                example = x[r]
            else:
                example = [x[r]]
        x_batch.append(example)
        x_length.append(len(x[r]))
        y_batch.append(y[r])

    assert len(x_batch) == len(y_batch) == len(x_length) == batch_size , ("failed batch")
    return np.array(x_batch), np.array(y_batch), np.array(x_length)




#return whole batch with respect to the order of x
def get_batch_fixed(x,y,max_time, PAD_ID):
    assert len(x) == len(y), ("X and Y dont have same size lel")

    x_batch, y_batch, x_length = [], [], []
    for i in range(len(x)):
        encoder_pad = [PAD_ID] * (max_time - len(x[i]))
        if len(x[i]) != max_time:
            if isinstance(PAD_ID,np.ndarray):
                example = (np.concatenate([x[i],encoder_pad]))
            else:
                example = ([x[i] + encoder_pad])
        else:
            if isinstance(PAD_ID,np.ndarray):
                example = x[i]
            else:
                example = [x[i]]
        x_batch.append(example)
        x_length.append(len(x[i]))
        y_batch.append(y[i])

    assert len(x_batch) == len(y_batch) == len(x_length) == len(x) , ("failed batch")
    return np.array(x_batch), np.array(y_batch), np.array(x_length)




def make_ids_corpus(dataset):
    with open(dataset + "vocab.en", "r") as f:
        l = f.readlines()
        voc = {}
        for i, r in enumerate(l):
            voc[i] = r.strip("\n")

    inv_voc = {v: k for k, v in voc.items()}
    with open(dataset + "train.ids.en", "w") as f:
        with open(dataset + "train.en", "r") as f2:
            l = f2.readlines()
            for r in l:
                new_line = [str(inv_voc[w]) for w in r.split()]
                f.write(' '.join(new_line))
                f.write("\n")
    return len(inv_voc)


