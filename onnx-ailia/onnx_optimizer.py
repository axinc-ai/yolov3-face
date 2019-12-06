import sys
import os
import argparse
import numpy
import onnx
import onnx.helper
import onnx.checker
import onnx.optimizer
import onnx.version_converter

SCRIPT_NAME = '+onnx_optimizer'
SCRIPT_VERSION = '+0.3'

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
    s_model = onnx.helper.make_model(s_graph, producer_name=model.producer_name, producer_version=model.producer_version, opset_imports=model.opset_import)

    return s_model

def convert_to_list(arg) :
    ret = []
    for e in arg :
        ret.append(e)
    return ret

def yolov3_special_treatment(model) : 

    ver = model.opset_import[0].version
    if (ver != 10) :
        detail = 'unsupported model opset_version=({})'.format(ver)
        raise Exception(detail)

    mod_input = convert_to_list(model.graph.input)

    erase_target = []

    # replace loop nodes & add input parameter
    mod_node = []
    for n in model.graph.node : 
        if (n.op_type == 'Loop') :
            erase_target.append( n.input[1] )
            erase_target.append( n.input[2] )
            wn_name = 'usq/' + n.input[0]
            ni = [ n.input[0] ]
            no = [ wn_name ] 
            nn = onnx.helper.make_node("Unsqueeze", ni, no, name=wn_name, axes=[0])
            mod_node.append( nn )
            ni = [ 'arange_base', 'arange_start', wn_name ]
            no = [ n.output[1] ]
            nn = onnx.helper.make_node("Slice", ni, no, name=n.name)
            mod_node.append( nn )
        elif (n.op_type == 'NonMaxSuppression') :
            iou = n.input[3]
            threshold = n.input[4]
            # workaround for AILIA
            erase_target.append( iou )
            erase_target.append( threshold )
            mod_input.append( onnx.helper.make_tensor_value_info(iou, onnx.TensorProto.FLOAT, (1,)) )
            mod_input.append( onnx.helper.make_tensor_value_info(threshold, onnx.TensorProto.FLOAT, (1,)) )
            mod_node.append( n )
        else :
            mod_node.append( n )

    # remove zero subtract
    mod2_node = []
    for n in mod_node : 
        if (n.op_type == 'Sub') and (n.input[1] in erase_target) :
            ni = [n.input[0]]
            no = [n.output[0]]
            nn = onnx.helper.make_node("Identity", ni, no, name=n.name)
            mod2_node.append( nn )
        else :
            mod2_node.append( n )
    
    mod_node = mod2_node

    # remove unused initializer element
    mod_initializer = []
    for e in model.graph.initializer :
        if not e.name in erase_target : 
            mod_initializer.append( e )
    
    # append arange_base (0..512 / INT32 / 1D Tensor)
    arange_base = numpy.arange( 0, 512, dtype=int )
    mod_initializer.append( onnx.helper.make_tensor('arange_base', onnx.TensorProto.INT32, [512], arange_base) )
    # append arange_start (0 / INT64 / 1D Tensor)
    arange_start = numpy.zeros( [1], dtype=numpy.int64 )
    mod_initializer.append( onnx.helper.make_tensor('arange_start', onnx.TensorProto.INT64, [1], arange_start) )

    m_graph = onnx.helper.make_graph(mod_node, model.graph.name, mod_input, model.graph.output, mod_initializer)
    m_model = onnx.helper.make_model(m_graph, producer_name=model.producer_name + SCRIPT_NAME, producer_version=model.producer_version+SCRIPT_VERSION, opset_imports=model.opset_import)

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

        # convert version
        model = onnx.version_converter.convert_version(model, 10)

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

