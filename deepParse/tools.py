import gzip
import os
import shutil

import torch.nn as nn
import torch.nn.init as init
from fasttext.util.util import valid_lang_ids, _download_gz_model


def download_fasttext_model(lang_id: str, saving_dir: str) -> str:
    """
    The source code of this function was copied from the fastText project, and has been modified.

    COPYRIGHT

    All contributions from the https://github.com/facebookresearch/fastText authors.
    Copyright (c) 2016 - August 13 2020
    All rights reserved.

    Each contributor holds copyright over their respective contributions. The project versioning (Git)
    records all such contribution source information.

    LICENSE

    The MIT License (MIT)

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial
    portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    """
        Simpler version of the download_model function from fastText to download pre-trained common-crawl
        vectors from fastText's website https://fasttext.cc/docs/en/crawl-vectors.html and save it in the
        saving directory (saving_dir).
    """
    if_exists = 'ignore'
    if lang_id not in valid_lang_ids:
        raise Exception("Invalid lang id. Please select among %s" %
                        repr(valid_lang_ids))

    file_name = os.path.join(saving_dir, "cc.%s.300.bin" % lang_id)
    gz_file_name = "%s.gz" % file_name

    if os.path.isfile(file_name):
        if if_exists == 'ignore':
            return file_name

    if _download_gz_model(gz_file_name, if_exists):
        with gzip.open(gz_file_name, 'rb') as f:
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f, f_out)

    return file_name


def weight_init(m):
    """
    Function to initialize the weight of a layer.
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
