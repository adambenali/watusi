import random
import os
import sys
import json
import argparse
import re
import pprint
import uuid
import lizard

# TODO: Replace immediate values with IMM1, IMM2 .. 
#       

EMCC = "emcc"
OBJDUMP = "llvm-objdump"
WASM2WAT = "wasm2wat"

DEFAULT_CONFIG = {
    "grammar": {
        "Statement": ["Assignment"],
        "Expression": ["Number"]
    },
    "max_nesting": 2,
    "max_nb_statements": 1,
    "max_nb_vars": 2,
    "max_call_params": 2,
    "max_functions": 1,
    "max_branch_statements": 2,
    "max_array_size": 10,
}

config = DEFAULT_CONFIG 

IRRELEVANT_CODE_INDICATOR = "__IRR__"

global_program = None
VAR_PREFIX = "VAR"

class Number:
    _nestable = False

    def __init__(self, nesting=None):
        self._generate()
    
    def _generate(self):
        self.value = random.randint(1, 100)
    
    def __str__(self):
        return str(self.value)
    
    def msk(self):
        return str(self.value)
        # return "IMM"

class String:
    pass

class BinaryOp:
    _ops = ["*", "+", "-", "/"]
    _nestable = True

    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting):
        self.lhs = Expression(nesting=nesting + 1)
        self.op = random.choice(self._ops)
        self.rhs = Expression(nesting=nesting + 1)
        if isinstance(self.lhs.element, Number):
            while isinstance(self.rhs.element, Number):
                self.rhs = Expression(nesting=nesting + 1)

    
    def __str__(self):
        return f"( {self.lhs} {self.op} {self.rhs} )"
    
    def msk(self):
        # if ((isinstance(self.lhs.element, Number)) and 
        #     (isinstance(self.rhs.element, Number))):
        #     return str(eval(str(self).replace(" / ", " // ")))
        return f"( {self.lhs.msk()} {self.op} {self.rhs.msk()} )"


class Variable:
    _nestable = False

    def __init__(self, name=None, nesting=0):
        self.generate(name)
    
    def generate(self, name):
        var_id = random.choice([x for x in range(0, config["max_nb_vars"]) if x != ArrayIndexing._current_index])
        self.name = f"{VAR_PREFIX}{var_id}" if name is None else name
    
    def __str__(self):
        return self.name
    
    def msk(self):
        return self.name
        # return f"{VAR_PREFIX}0"

class IfBranch:
    _nestable = True

    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting=0):
        self.condition = Condition()
        self.if_body = self._generate_body(nesting + 1)
        self.else_body = self._generate_body(nesting + 1) if random.randint(0, 1) else None

    def _generate_body(self, nesting):
        return StatementList(max_statements=config["max_branch_statements"], nesting=nesting, types=["Assignment"])

    def __str__(self):
        if_clause = "\n".join([f"if {self.condition} {{", f"{self.if_body}", "}"])
        else_clause = "" if self.else_body is None else (
            "\n".join([" else {", f"{self.else_body}", "}"])
        )
        return if_clause + else_clause
    
    def msk(self):
        if_clause = "\n".join([f"if {self.condition.msk()} {{", f"{self.if_body.msk()}", "}"])
        else_clause = "" if self.else_body is None else (
            "\n".join([" else {", f"{self.else_body.msk()}", "}"])
        )
        return if_clause + else_clause

class SwitchBranch:
    pass

class WhileLoop:
    _nestable = True

    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting):
        self.condition = Condition()
        self.body = self._generate_body(nesting + 1)
    
    def _generate_body(self, nesting):
        # return StatementList(max_statements=config["max_branch_statements"], nesting=nesting)
        return StatementList(max_statements=2, nesting=nesting)
    
    def __str__(self):
        return "\n".join([f"while {self.condition} {{", str(self.body), "}"])
    
    def msk(self):
        return "\n".join([f"while {self.condition.msk()} {{", self.body.msk(), "}"])

class ForLoop:
    pass

class ArrayIndexing:
    _nestable = True
    _current_index = -1
    
    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting):
        self.name = f"VAR{ArrayIndexing._current_index}"
        self.index = Expression(nesting + 1)
        while isinstance(self.index.element, Number) and self.index.element.value >= 10:
            self.index = Expression(nesting + 1)
    def __str__(self):
        return f"{self.name} [ {self.index} ]"
    
    def msk(self):
        return f"{self.name} [ {self.index} ]"

class FunctionDeclaration:
    """
    (int | void) func0(int X1, .., int Xn)
    """
    def __init__(self):
        self._generate()
    
    def _generate(self):
        self.name = "func0"
        self.return_type = random.choice(FunctionCall._return_types)
        self.nb_args = random.randint(1, config["max_call_params"])

    def __str__(self):
        args = ', '.join(f"int X{i}" for i in range(self.nb_args))
        return f"{self.return_type} {self.name} ( {args} ) ;"
    
    def msk(self):
        args = ', '.join(f"int X{i}" for i in range(self.nb_args))
        return f"{self.return_type} {self.name} ( {args} ) ;"

class FunctionCall:
    _nestable = True
    _return_types = ["void", "int"]

    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting):
        prototype = random.choice(global_program.func_declarations)
        self.name = prototype.name
        self.return_type = prototype.return_type
        nb_params = prototype.nb_args
        self.params = [Expression(nesting=nesting + 1) for _ in range(nb_params)]
    
    def __str__(self):
        return f"{self.name} ( " + " , ".join(str(param) for param in self.params) + " )"
    
    def msk(self):
        return f"{self.name} ( " + " , ".join(param.msk() for param in self.params) + " )"

class FunctionCallStatement(FunctionCall):
    
    def __init__(self, nesting=0):
        super().__init__()
    
    def __str__(self):
        return super().__str__() + " ;"
    
    def msk(self):
        return super().msk() + " ;"

class Condition:
    _ops = [">", ">=", "<", "<=", "==", "!="]
    _nestable = True

    def __init__(self, nesting=0):
        self._generate(nesting)
    
    def _generate(self, nesting):
        self.lhs = Expression(nesting + 1)
        while isinstance(self.lhs.element, Number):
            self.lhs = Expression(nesting + 1)
        self.op = random.choice(self._ops)
        self.rhs = Expression(nesting + 1)

    def __str__(self):
        return f"( {self.lhs} {self.op} {self.rhs} )"
    
    def msk(self):
        return f"( {self.lhs.msk()} {self.op} {self.rhs.msk()} )"


class Expression:
    _types = [getattr(sys.modules[__name__], cls) for cls in config["grammar"]["Expression"]]
    _stop_types = list(filter(lambda cls: not cls._nestable, _types))

    def __init__(self, nesting=0):
        self.element = self._generate(nesting)
    
    def _generate(self, nesting=0):
        type_class = random.choice(
            self._types if nesting < config.get("max_nesting") else
            self._stop_types
        )
        return type_class(nesting=nesting)
    
    def __str__(self):
        return str(self.element)

    def msk(self):
        return self.element.msk()


class Assignment:
    _nestable = False

    def __init__(self, var=None, nesting=0):
        self._generate(var)

    def _generate(self, var):
        lhs_choices = [Variable(name=var)]
        if "ArrayIndexing" in config["grammar"]["Expression"]:
            lhs_choices += [ArrayIndexing()]
        self.lhs = random.choice(lhs_choices)
        self.rhs = Expression()
    
    def __str__(self):
        return f"{self.lhs} = {self.rhs} ;"
    
    def msk(self):
        return f"{self.lhs.msk()} = {self.rhs.msk()} ;"

class Declaration:
    _nestable = False
    def __init__(self, var=None, initialize=True, nesting=0):
        self._generate(var, initialize)

    def _generate(self, var, initialize):
        self.type = "int" # TODO: Generify this
        self.lhs = Variable(name=var)
        self.rhs = Number() if initialize else None
    
    def __str__(self):
        return f"{self.type} {self.lhs}" + (f" = {self.rhs} ;" if self.rhs else "")
    
    def msk(self):
        return f"{self.type} {self.lhs.msk()}" + (f" = {self.rhs.msk()} ;" if self.rhs else "")

class Statement:
    _types = [getattr(sys.modules[__name__], cls) for cls in config["grammar"]["Statement"]]
    _stop_types = list(filter(lambda cls: not cls._nestable, _types))

    def __init__(self, nesting=0):
        self.element = self._generate(nesting)
    
    def _generate(self, nesting):
        type_class = random.choice(
            self._types if nesting < config.get("max_nesting") else
            self._stop_types
        )
        return type_class(nesting=nesting)

    def __str__(self):
        return str(self.element)

    def msk(self):
        return self.element.msk()
    
class StatementList:
    def __init__(self, max_statements, nesting=0, types=None):
        self.max_statements = max_statements
        self._generate(nesting, types)
    
    def _generate(self, nesting, types):
        if types is not None:
            self.statements = [Assignment() for _ in range(random.randint(1, self.max_statements))]
        self.statements = [Statement(nesting=nesting) for _ in range(random.randint(1, self.max_statements))]

    def __str__(self):
        return "\n".join(map(str, self.statements))
    
    def msk(self):
        return "\n".join(map(lambda st: st.msk(), self.statements))


class Program:
    _masked_c = "main.masked.c"
    _original_c = "main.c"
    _original_c_backup = "main.bak.c"
    _wasm = "main.wasm"
    _wat = "main.wat"
    _objdump = "main.objdump"
    _compilation_folder = "/tmp/watusi/"
    _failure_folder = "failed"

    def __init__(self, nb_statements=None):
        global global_program
        global_program = self
        if "ArrayIndexing" in config["grammar"]["Expression"]:
            ArrayIndexing._current_index = random.randint(0, config["max_nb_vars"])
        if nb_statements is None:
            nb_statements = random.randint(1, config.get("max_nb_statements"))
        self.func_declarations = [FunctionDeclaration()]
        self.statements = [Statement() for _ in range(nb_statements)]
    
    def write(self):
        c_path = os.path.join(self._compilation_folder, self._original_c)
        c_file_content = self.generate_full_program(nb_vars=config["max_nb_vars"]) # TODO: Improve this
        with open(c_path, "w") as c_file:
            c_file.write(c_file_content)

        masked_c_file_content = self.generate_full_program(mask=True, nb_vars=config["max_nb_vars"])
        masked_c_path = os.path.join(self._compilation_folder, self._masked_c)
        with open(masked_c_path, "w") as c_file:
            c_file.write(masked_c_file_content)
    
    def generate_full_program(self, mask=False, globals="", func_name="main", func_args=["int argc", "char* argv[]"], nb_vars=1, init_value=0x0000BEEF):
        # var_list = "int " + ", ".join(f"X{i} = {init_value}" for i in range(nb_vars)) + ";\n"
        func_declarations = "\n".join(
            declaration.msk() if mask else str(declaration)
            for declaration in self.func_declarations
        ) + "\n"
        var_order = [i for i in range(nb_vars) if i != ArrayIndexing._current_index]
        random.shuffle(var_order)
        array_initialization = ', '.join(str(random.randint(0, 100)) for _ in range(config["max_array_size"]))
        var_list = "\n".join(
            (Declaration(f"{VAR_PREFIX}{i}").msk() if mask else str(Declaration(f"{VAR_PREFIX}{i}"))) +
            " // " + IRRELEVANT_CODE_INDICATOR
            for i in var_order
        ) + "\n"
        if "ArrayIndexing" in config["grammar"]["Expression"]:
            var_list += f"int VAR{ArrayIndexing._current_index} [ ] = {{{array_initialization}}}; // {IRRELEVANT_CODE_INDICATOR}" + "\n"
        func_args_string = ", ".join(func_args)
        main_declaration = f"int {func_name}({func_args_string}) {{ // {IRRELEVANT_CODE_INDICATOR}" + "\n"
        body = (str(self) if not mask else self.msk()) + "\n"
        return (globals +
                func_declarations +
                main_declaration +
                var_list +
                body +
                "}\n")

    def compile(self):
        wasm_path = os.path.join(self._compilation_folder, self._wasm)
        c_path = os.path.join(self._compilation_folder, self._original_c)
        masked_c_path = os.path.join(self._compilation_folder, self._masked_c)
        backed_c_path = os.path.join(self._compilation_folder, self._original_c_backup)
        compile_args = '-g -s WASM=1 -s SIDE_MODULE=1 -Wl,--export-all -Wl,--allow-undefined -O0'
        ret = os.system(f"{EMCC} {compile_args} -o {wasm_path} {c_path} > /tmp/LOG 2>&1")
        os.system(f"cp {c_path} {backed_c_path}")
        os.system(f"cp {masked_c_path} {c_path}")

        if ret != 0:
            failed_path = os.path.join(self._compilation_folder, self._failure_folder, str(uuid.uuid4()) + ".c")
            c_backup_path = os.path.join(self._compilation_folder, self._original_c_backup)
            os.system(f"cp {c_backup_path} {failed_path}")
        
        return ret
    
    @classmethod
    def compile_one(cls, c_path):
        wasm_path = os.path.join(cls._compilation_folder, cls._wasm)
        compile_args = '-g -s WASM=1 -s SIDE_MODULE=1 -Wl,--export-all -Wl,--allow-undefined -O0'
        ret = os.system(f"{EMCC} {compile_args} -o {wasm_path} {c_path} > /tmp/LOG 2>&1")

        wat_path = os.path.join(cls._compilation_folder, cls._wat)
        command_args = '-f'
        os.system(f"{WASM2WAT} {command_args} -o {wat_path} {wasm_path} >> /tmp/LOG 2>&1")

        objdump_path = os.path.join(cls._compilation_folder, cls._objdump)
        command_args = "-S --no-show-raw-insn --no-leading-addr"
        os.system(f"{OBJDUMP} {command_args} {wasm_path} > {objdump_path}")

    @staticmethod
    def get_mappings_from_objdump(path):
        with open(path, "r") as objdump_file:
            lines = objdump_file.readlines()
            source_code_lines = [i for i in range(len(lines)) if lines[i][0] == ';'] + [len(lines)]
            mappings = {}
            for i in range(len(source_code_lines) - 1):
                current_line_index = source_code_lines[i]
                next_line_index = source_code_lines[i + 1]
                stripped_hl = re.sub(r"\s+", " ", lines[current_line_index][1:]).strip()
                stripped_ll = (
                    " ".join(re.sub(r"\s+", " ", line).strip() 
                    for line in lines[current_line_index + 1: next_line_index]).strip()
                )
                mappings[stripped_hl] = stripped_ll
        
        return mappings
    
    @staticmethod
    def get_full_program_from_objdump(path):
        with open(path, "r") as objdump_file:
            lines = objdump_file.readlines()
            relevant_code_start = 0
            relevant_code_end = len(lines) - 1
            for i, line in enumerate(lines):
                if line.strip().startswith(";") and IRRELEVANT_CODE_INDICATOR not in line:
                    relevant_code_start = i + 1
                    break
            else:
                assert False, ("Could not find the start of the relevant part of the code")
            
            for i, line in enumerate(lines):
                if re.sub(r"\s+", " ", line).strip() == "; }":
                    relevant_code_end = i - 1

            wasm_code_lines = [lines[i] for i in range(relevant_code_start, relevant_code_end + 1) if lines[i][0] != ";"]
            stripped_ll = " ".join(re.sub(r"\s+", " ", line).strip() for line in wasm_code_lines).strip()
        
            return stripped_ll
    
    @staticmethod
    def get_full_program_from_wat(path):
        pass
    
    @staticmethod
    def replace_immediate_values(identifier, low_level, high_level):
        ll_imm_values = list({int(match[10:]) for match in re.findall("i32.const \d+", low_level)})
        ll_imm_values = [val for val in ll_imm_values if val > 9] # Only replace values > 9
        hl_imm_values = list({int(val) for val in re.findall("\d+", high_level) if int(val) > 9})
        
        ll_imm_values.sort()
        hl_imm_values.sort()
        replacements = {}
        if ll_imm_values != hl_imm_values:
            print("Warning ! Immediate values differ @", identifier)
        
        for i, value in enumerate(ll_imm_values):
            replacements[value] = f"IMM{i + 1}"
            low_level = low_level.replace(f"i32.const {value}", f"i32.const IMM{i + 1}")
        
        for i, value in enumerate(hl_imm_values):
            high_level = high_level.replace(f" {value} ", f" {replacements[value]} " if value in replacements else f" {value} ")

        return low_level, high_level, replacements
        


    def wasm_to_wat(self):
        wasm_path = os.path.join(self._compilation_folder, self._wasm)
        wat_path = os.path.join(self._compilation_folder, self._wat)
        command_args = '' #'-f'
        os.system(f"{WASM2WAT} {command_args} -o {wat_path} {wasm_path} >> /tmp/LOG 2>&1")

    def llvm_objdump(self):
        wasm_path = os.path.join(self._compilation_folder, self._wasm)
        objdump_path = os.path.join(self._compilation_folder, self._objdump)
        command_args = "-S --no-show-raw-insn --no-leading-addr"
        ret = os.system(f"{OBJDUMP} {command_args} {wasm_path} > {objdump_path}")
        if ret != 0:
            failed_path = os.path.join(self._compilation_folder, self._failure_folder, str(uuid.uuid4()) + ".c")
            c_backup_path = os.path.join(self._compilation_folder, self._original_c_backup)
            os.system(f"cp {c_backup_path} {failed_path}")
        
        return ret
    
    def __str__(self):
        return '\n'.join(map(str, self.statements))

    def msk(self):
        return '\n'.join([st.msk() for st in self.statements])


def apply_config(config):
    for node in config["grammar"]:
        node_class = getattr(sys.modules[__name__], node)
        node_class._types = [getattr(sys.modules[__name__], cls) for cls in config["grammar"][node]]
        try:
            old_value = node_class._stop_types
            node_class._stop_types = list(filter(lambda cls: not cls._nestable, node_class._types))
        except:
            pass


def generate_dataset(count, unique=True, out=None, chunks=False):
    if out is None:
        out = "out"
    
    with open(f"data/{out}.meta", "w") as meta:
        with open(f"data/{out}.ll", "w") as ll:
            with open(f"data/{out}.hl", "w") as hl:
                high_level_corpus = []
                low_level_corpus = []
                meta_corpus = []
                total = 0
                generated_pairs = set()
                nb_programs = 0
                average_cc = 0 # Cyclomatic complexity

                while nb_programs < count:
                    if nb_programs > 0 and nb_programs % 100 == 0:
                        ll.write("\n".join(low_level_corpus) + "\n")
                        hl.write("\n".join(high_level_corpus) + "\n")
                        high_level_corpus = []
                        low_level_corpus = []
                    nb_programs += 1
                    print(f"{nb_programs} / {count}\r", end="", flush=True)
                    p = Program()
                    p.write()
                    failed_compilation = p.compile()
                    if failed_compilation:
                        continue
                    p.wasm_to_wat()
                    failed_objdump = p.llvm_objdump()
                    if failed_objdump:
                        continue
                    
                    meta_dict = {}
                        
                    if chunks:
                        mapping = Program.get_mappings_from_objdump(os.path.join(Program._compilation_folder, Program._objdump))
                        for high_level, low_level in mapping.items():

                            if unique and (high_level, low_level) in generated_pairs:
                                continue

                            if high_level.endswith(IRRELEVANT_CODE_INDICATOR):
                                continue

                            generated_pairs.add((high_level, low_level))

                            low_level, high_level, replacements = Program.replace_immediate_values(total + 1, low_level, high_level)

                            original_source_code = str(p).replace("\n", " ")
                            meta_dict["replacements"] = replacements
                            meta_dict["original_sc"] = original_source_code
                            meta_dict["complexity"] = lizard.analyze_file.analyze_source_code(
                                "main.c", 
                                f"int main(){{{original_source_code}}}"
                            ).function_list[0].cyclomatic_complexity

                            average_cc += meta_dict["complexity"]

                            high_level_corpus.append(high_level)
                            low_level_corpus.append(low_level)
                            meta_corpus.append(str(meta_dict))

                            total += 1
                    else:
                        try:
                            low_level = Program.get_full_program_from_objdump(os.path.join(Program._compilation_folder, Program._objdump))
                            high_level = p.msk().replace("\n", " ")
                            low_level, high_level, replacements = Program.replace_immediate_values(total + 1, low_level, high_level)
                            
                            original_source_code = str(p).replace("\n", " ")
                            meta_dict["replacements"] = replacements
                            meta_dict["original_sc"] = original_source_code
                            meta_dict["complexity"] = lizard.analyze_file.analyze_source_code(
                                "main.c", 
                                f"int main(){{{original_source_code}}}"
                            ).function_list[0].cyclomatic_complexity

                            average_cc += meta_dict["complexity"]

                            high_level_corpus.append(high_level)
                            low_level_corpus.append(low_level)
                            meta_corpus.append(str(meta_dict))

                            total += 1
                        except AssertionError:
                            pass

                print()
                average_cc /= total
                meta_corpus.append(f"Average cyclomatic complexity \t {average_cc}\n")
                meta_corpus.append(f"seed \t {seed}\n")
                meta.write("\n".join(meta_corpus) + "\n")
                ll.write("\n".join(low_level_corpus) + "\n")
                hl.write("\n".join(high_level_corpus) + "\n")

    
    # Write dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate C snippets")
    parser.add_argument('-p', '--program', dest='p', type=str, help="Compile a particular program")
    parser.add_argument('-c', '--config', dest='config', type=str, help="Configuration")
    parser.add_argument('-s', '--seed', dest='seed', type=int, help="Seed")
    parser.add_argument('-n', '--count', dest='count', type=int, help="Number of programs to generate", default=1)
    parser.add_argument('-o', '--out', dest='out', type=str, help="Output file")
    parser.add_argument('--chunks', dest='chunks', action="store_true", help="Divide program into chunks")
    parser.add_argument('--cf', dest='compilation_folder', help="Compilation folder")
    args = parser.parse_args()
    
    seed = args.seed or random.randint(0, 1 << 32)
    random.seed(seed)
    print(f"Seed: {seed}")
    
    if args.compilation_folder is not None:
        Program._compilation_folder = args.compilation_folder

    # Init temp folders
    if not os.path.exists(Program._compilation_folder):
        os.mkdir(Program._compilation_folder)
    if not os.path.exists(os.path.join(Program._compilation_folder, Program._failure_folder)):
        os.mkdir(os.path.join(Program._compilation_folder, Program._failure_folder))

    if args.p is not None:
        Program.compile_one(args.p)
        exit(0)
    
    if args.config:
        with open(args.config) as config_file:
            config.update(json.load(config_file))

    pprint.pprint(config)
    apply_config(config)
    generate_dataset(args.count, out=args.out, chunks=args.chunks)