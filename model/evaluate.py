from difflib import SequenceMatcher
import sys
import io
from pycparser import c_parser, c_ast, parse_file, plyparser
import lizard

def same(x, y):
    """
    returns if tokens are same structure
    """

    return x == y or (x.startswith("VAR") and y.startswith("VAR")) or (x.startswith("IMM") and y.startswith("IMM"))


def compare(p, q, d=0):
    # print("\t" * d + "<---")
    # s = io.StringIO()
    # p.show(s)
    # s.write("\n")
    # q.show(s)
    # print("\n".join("\t" * d + line for line in s.getvalue().split("\n")))
    # print("\t" * d + "--->")

    lchildren = p.children()
    rchildren = q.children()
    # if len(lchildren) != len(rchildren):
    #     return 0
    score = 0   
    size = 0
    if not p.children() and not q.children():
        # print(f" -- End nodes -- ")
        # p.show()
        # q.show()
        # print("-- \t \t --")
        return int(type(p) == type(q)), 1
    elif p.children() and not q.children():
        return 0, 1
    elif not p.children() and q.children():
        _, size = compare(q, q, d + 1)
        return 0, size
    if isinstance(p, c_ast.BinaryOp) and isinstance(q, c_ast.BinaryOp) and q.op in ["==", "+", "*"]:
        v11, size11 = compare(p.left, q.left, d + 1)
        v12, size12 = compare(p.right, q.right, d + 1)
        v21, size21 = compare(p.left, q.right, d + 1)
        v22, size22 = compare(p.right, q.left, d + 1)
        same_op = int(p.op == q.op)
        score = (same_op + max((v11 * size11 + v12 * size12), (v21 * size21 + v22 * size22))) / (size11 + size12 + 1)
        return score, size11 + size12 + 1

    for i in range(max(len(rchildren), len(lchildren))):
        left_child = lchildren[i][1] if i < len(lchildren) else c_ast.Continue() # Something that will never equal rchildren_i
        right_child = rchildren[i][1] if i < len(rchildren) else c_ast.Continue() # Something that will never equal rchildren_i
        part_score, part_size = compare(left_child, right_child, d + 1)
        score += part_score * part_size
        size += part_size
        # print('\t' * d + f"@ {d}", part_score, part_size)
    
    if type(q) in [c_ast.If, c_ast.BinaryOp, c_ast.While, c_ast.Assignment, c_ast.FuncCall, c_ast.ArrayRef]:
        score += (type(p) == type(q))
        size += 1
    # else:
        # s = io.StringIO()
        # q.show(s)
        # print(s.getvalue())
        # print(type(q))
    score /= size

    # print("\t" * d + str(score), size)
    
    return score, size


folder = "run" if len(sys.argv) == 1 else sys.argv[1]

prediction = open(f"{folder}/run/test_prediction", "r").readlines()
ground_truth = open(f"{folder}/data/test.hl", "r").readlines()

average_acc = 0
predictions = 0

# Length, CC, token accuracy, template accuracy
results = []

for i, (pred, target) in enumerate(zip(prediction, ground_truth)):
    try:
        ast_source = c_parser.CParser().parse("int main() {" + pred + "}")
    except Exception as e:
        # try:
        #     ast_source = c_parser.CParser().parse("int main() {" + pred.strip()[:-1] + "}")
        # except:
        #     print(f"Warning! Unparsable @ {i + 1}", e, pred)
        #     continue
        continue
    ast_target = c_parser.CParser().parse("int main() {" + target + "}")
    acc, nb_tokens = compare(ast_source.children()[0][1].body, ast_target.children()[0][1].body)
    template_acc = acc

    cc = lizard.analyze_file.analyze_source_code("main.c", f"int main(){{{target.strip()}}}" ).function_list[0].cyclomatic_complexity

    # print(i + 1, acc)
    pred = pred.split()
    target = target.split()
    if len(pred) == 0:
        acc = 0
    else:
        acc = sum(same(x, y) for x, y in zip(pred, target)) / min(len(pred), len(target))
    token_acc = acc

    # m = SequenceMatcher(None, pred, target)
    # acc = m.ratio()
    # print(acc)
    average_acc += acc
    predictions += 1
    length = len(target)

    results.append([length, cc, token_acc, template_acc])

results.sort(key=lambda res: res[0])
[print(item) for item in results]

average_acc /= predictions

print("Parsable predictions:", predictions)
print("Average accuracy:", average_acc)