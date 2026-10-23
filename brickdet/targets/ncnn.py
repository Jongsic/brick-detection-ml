"""ncnn via pnnx (fp16 weights): <out>.param, <out>.bin"""
import os
import shutil
import tempfile

import numpy as np

IO = {'input': 'in0', 'output': 'out0'}


def export(net, x, out):
    import pnnx
    tmp = tempfile.mkdtemp()
    try:
        pnnx.export(net, os.path.join(tmp, 'model.pt'), (x,))
        for ext in ('.param', '.bin'):
            shutil.copy(os.path.join(tmp, 'model.ncnn' + ext), out + ext)
    finally:
        shutil.rmtree(tmp)
    return [out + '.param', out + '.bin']


def load(out):
    import ncnn
    net = ncnn.Net()
    net.load_param(out + '.param')
    net.load_model(out + '.bin')

    def run(x):
        ex = net.create_extractor()
        ex.input(IO['input'], ncnn.Mat(np.ascontiguousarray(x[0])))
        _, y = ex.extract(IO['output'])
        return np.array(y)[None]

    return run
