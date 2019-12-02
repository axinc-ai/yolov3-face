import sys
import os
import argparse
import numpy
import onnx
import onnx.helper
import onnx.checker
import onnx.optimizer

class NodeInfo :
    def __init__(self, node) :
        self.__node = node
        self.__input = []
        self.__output = []
        for s in node.input :
            self.__input.append(s)
        for s in node.output : 
            self.__output.append(s)

    def clear_input(self, name) :
        while self.__input.count(name) > 0 :
            self.__input.remove(name)
        return len(self.__input)

    def check_resolve(self) :
        return (len(self.__input) == 0)

    def ref_output(self) : 
        return self.__output

    def ref_node(self) : 
        return self.__node

def onnx_topologically_sort(model) :
    pool = []
    for e in model.graph.node : 
        pool.append(NodeInfo(e))

    for e in model.graph.initializer :
        name = e.name
        for i in range(len(pool)) :
            pool[i].clear_input(name)

    for e in model.graph.input :
        name = e.name 
        for i in range(len(pool)) :
            pool[i].clear_input(name)

    sorted = []
    while len(pool) > 0 :
        removed = []
        removed_name = []
        for i in range(len(pool)) :
            if pool[i].check_resolve() :
                removed.append(i)
                removed_name.extend(pool[i].ref_output())

        removed.reverse()
        for i in removed :
            sorted.append(pool[i].ref_node())
            pool.pop(i)

        for name in removed_name :
            for n in pool : 
                n.clear_input(name)

    s_graph = onnx.helper.make_graph(sorted, model.graph.name, model.graph.input, model.graph.output, model.graph.initializer)
    s_model = onnx.helper.make_model(s_graph, producer_name=model.producer_name, producer_version=model.producer_version)

    return s_model

def yolov3_special_treatment(model) : 

    # append arange_base (0..1024 / INT32 / 1D Tensor)
    arange_base = numpy.arange( 0, 1024, dtype=int )
    mod_initializer = model.graph.initializer
    mod_initializer.append( onnx.helper.make_tensor('arange_base', onnx.TensorProto.INT64, [1024], arange_base) )

    mod_input = model.graph.input

    # replace loop nodes
    mod_node = []
    for n in model.graph.node : 
        if (n.op_type == 'Loop') :
            ni = [ 'arange_base', n.input[2], n.input[0] ]
            no = [ n.output[1] ]
            nn = onnx.helper.make_node("Slice", ni, no, name=n.name)
            mod_node.append( nn )
        elif (n.op_type == 'NonMaxSuppression') :
            iou = n.input[3]
            threshold = n.input[4]
            mod_input.append( onnx.helper.make_tensor_value_info(iou, onnx.TensorProto.FLOAT, None) )
            mod_input.append( onnx.helper.make_tensor_value_info(threshold, onnx.TensorProto.FLOAT, None) )
            mod_node.append( n )
        else :
            mod_node.append( n )

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name + '+onnx_optimizer', producer_version=model.producer_version+'+0.2')

    return m_model


def onnx_optimize(onnx_path, yolov3_mode) : 

    # show information
    out_dir = os.path.dirname(onnx_path)
    out_base, out_ext = os.path.splitext(os.path.basename(onnx_path))
    out_path = out_base + ".opt" + out_ext
    if out_dir != '' :
        out_path = out_dir + os.path.sep + out_path
    print("+ creating " + out_path)
    print("    from " + onnx_path + " ...")

    # load model
    model = onnx.load(onnx_path)

    # topologically sort
    model = onnx_topologically_sort(model)

    if yolov3_mode :
        # yolov3 : optimization failed / do loop replacement + add input (NonMaxSuppression threshold&iou)
        model = yolov3_special_treatment(model)
    else :
        # precheck
        onnx.checker.check_model(model)

        # optimize 
        # model = onnx.optimizer.optimize(model, onnx.optimizer.get_available_passes() )
        opt_passes = [
            'extract_constant_to_initializer', 
            'fuse_add_bias_into_conv', 
            'fuse_bn_into_conv', 
            'fuse_consecutive_concats', 
            'fuse_consecutive_log_softmax', 
            'fuse_consecutive_reduce_unsqueeze', 
            'fuse_consecutive_squeezes', 
            'fuse_consecutive_transposes', 
            'fuse_matmul_add_bias_into_gemm', 
            'fuse_pad_into_conv', 
            'fuse_transpose_into_gemm', 
        ]
        model = onnx.optimizer.optimize(model, opt_passes )

        # following passes are not worked (output broken model)
        # opt_eliminate_passes = [
        #    'eliminate_deadend', 
        #    'eliminate_identity', 
        #    'eliminate_nop_dropout', 
        #    'eliminate_nop_monotone_argmax', 
        #    'eliminate_nop_pad', 
        #    'eliminate_nop_transpose', 
        #     'eliminate_unused_initializer', 
        # ]
        # model = onnx.optimizer.optimize(model, opt_eliminate_passes )

    # save result
    with open(out_path, "wb") as f:
        f.write(model.SerializeToString())


def main() :
    parser = argparse.ArgumentParser( description='ONNX optimizer' )
    parser.add_argument('files', nargs='+', help='optimize target file.')
    parser.add_argument('--yolov3', action='store_true', help='run YOLOv3(from keras) model special mode.\n')
    args = parser.parse_args()

    for file in args.files :
        onnx_optimize(file, args.yolov3)


if __name__ == "__main__":
    main()

