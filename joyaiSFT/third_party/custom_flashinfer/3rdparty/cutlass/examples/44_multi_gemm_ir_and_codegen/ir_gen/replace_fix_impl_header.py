import os

class replace_fix_impl:

    def __init__(self, src_dir, dst_dir, cutlass_deps_root):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.cutlass_deps_root = cutlass_deps_root

    def gen_code(self):
        for sub_dir in os.walk(self.src_dir):
            files_in_sub_dir = sub_dir[2]
            src_dirs = sub_dir[0]
            output_dirs = self.dst_dir + sub_dir[0][len(self.src_dir):]
            if not os.path.exists(output_dirs):
                os.mkdir(output_dirs)
            for f in files_in_sub_dir:
                with open(src_dirs + '/' + f, 'r') as current_file:
                    output_lines = []
                    lines = current_file.readlines()
                    for line in lines:
                        if len(line) >= len('#include "cutlass') and line[:len('#include "cutlass')] == '#include "cutlass':
                            new_line = '#include "' + self.cutlass_deps_root + line[len('#include "'):]
                            output_lines.append(new_line)
                        else:
                            output_lines.append(line)
                    with open(output_dirs + '/' + f, 'w+') as dest_file:
                        dest_file.writelines(output_lines)