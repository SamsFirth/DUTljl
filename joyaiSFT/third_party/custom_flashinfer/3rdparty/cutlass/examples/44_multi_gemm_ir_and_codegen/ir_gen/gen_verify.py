import helper
import gen_ir as ir
import gen_turing_and_volta as gen_basic

class gen_verify:

    def __init__(self, fuse_gemm_info, gen_class_name, user_header_file, output_dir='../'):
        self.fuse_gemm_info = fuse_gemm_info
        self.name = gen_class_name + '_verify'
        self.b2b_num = len(fuse_gemm_info)
        self.params = []
        self.user_header_file = ''
        for header in user_header_file:
            self.user_header_file += '#include "' + header + '"\n'
        self.separate_cutlass = gen_basic.gen_volta_turing_fuse_act_impl(fuse_gemm_info, gen_class_name, user_header_file, output_dir)
        self.gen_params()
        self.output_dir = output_dir

    def gen_code(self):
        code = ''
        code += self.user_header_file
        code += self.separate_cutlass.gen_using(False)
        code_body = ''
        for i in range(self.b2b_num):
            code_body += '    ' + helper.var_idx('Gemm', i) + helper.var_idx(' gemm_op_', i) + ';\n'
            code_body += '    ' + helper.var_idx('gemm_op_', i) + helper.var_idx('.initialize(Arguments_', i) + ', nullptr);\n'
        code_body += self.separate_cutlass.gen_run()
        code += ir.gen_func(self.name, self.params, code_body)
        helper.write_2_headfile('cutlass_verify.h', self.output_dir, code)

    def gen_params(self):
        for i in range(self.b2b_num):
            self.params.append((helper.var_idx('typename Gemm', i) + '::Arguments', helper.var_idx('Arguments_', i)))

    def get_params(self, declartion=True):
        code = ''
        if declartion:
            for param in self.params:
                code += param[0] + ' ' + param[1] + ';\n'
        return code

    def gen_initialize():
        code = ''
        initialize_code = self.separate_cutlass.gen_initialize()
        code = ir.gen_func('initialize', [[]])