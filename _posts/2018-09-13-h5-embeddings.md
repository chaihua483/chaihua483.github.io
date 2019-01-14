---
layout: post
title:  "使用 HDF5 存储 Word Embedding"
date:   2018-09-13 15:40:56
categories: tips
tags: 
---

## 背景

估计有很多同学已经用hdf5存储 word vectors了，不过今天看到周围的小伙伴依然在抱怨加载个 word2vec 需要等5分钟，这对于在工作站上训练模型的同学来说是个很讨厌的事儿，所以本文直接上干货，提供个现成的class直接完成 gensim keyedvectors 到 hdf5 格式的转换，用 c++ 的找 hdf5.h 实现加载。

关于HDF5的介绍：[什么是hdf5](https://support.hdfgroup.org/HDF5/)

## 性能对比

看下同样的数据加载时间的区别：

数据源：facebook提供的中文维基百科词向量训练结果。矩阵大小：332647\*300，文本文件大小：822M。

![](/images/blog-h5/1.png)

大概50倍的样子。


## 实现过程

话不多说，直接上代码，反正没几行，只在python2.7下测试通过，python3的话修改一下有关编码的部分应该就ok。

```python
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import Vocab
from gensim import utils


class H5KeyedVectors(KeyedVectors):

    @classmethod
    def load_h5_format(cls, path, fvocab=None):

        # load counts
        counts = None
        if fvocab is not None:
            counts = {}
            with utils.smart_open(fvocab) as fin:
                for line in fin:
                    word, count = utils.to_unicode(line).strip().split()
                    counts[word] = int(count)
        # read h5
        import h5py
        with h5py.File(path, "r") as f:
            words = map(lambda x: x.decode("utf8"), f["words"].value.tolist())
            vecs = f["vecs"].value
        assert len(words) == vecs.shape[0]
        assert len(vecs.shape) == 2

        vocab_size, vector_size = vecs.shape
        result = cls(vector_size)
        result.vector_size = vector_size
        result.vectors = vecs
        if counts is None:
            result.vocab = {w: Vocab(index=i, count=1) for i, w in enumerate(words)}
        else:
            result.vocab = {w: Vocab(index=i, count=counts.get(w, 1)) for i, w in enumerate(words)}

        assert vocab_size == len(result.vocab)
        result.index2word = words
        return result

    def save_h5_format(self, path):
        arr_dict = {
            "words": map(lambda x: x.encode("utf8"), self.index2word),
            "vecs": self.vectors
        }
        import h5py
        with h5py.File(path, "w") as h5:
            for k, v in arr_dict.iteritems():
                if isinstance(v[0], basestring):
                    h5.create_dataset(k, data=v, dtype=h5py.special_dtype(vlen=str))
                else:
                    h5.create_dataset(k, data=v, dtype=v.dtype)
```


转化方法很简单:

```python
path = "./test.vec"
out_path = "./test.h5"

keyed_vectors = H5KeyedVectors.load_word2vec_format(path)

# save h5
keyed_vectors.save_h5_format(out_path)

# load h5
keyed_vectors_new = H5KeyedVectors.load_h5_format(out_path)
```

需要注意的是 python2 的 unicode 在hdf5 下存储的时候会各种问题，建议转化为 utf8 编码，并且使用 vlen 的方式存储。
