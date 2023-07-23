"""实现简单的四则运算表达式计算器"""

import re


def simple_calculate(operand1: float, operand2: float, operator: str) -> float:
    """四则运算"""
    if operator == "+":
        return operand1 + operand2
    if operator == "-":
        return operand1 - operand2
    if operator == "*":
        return operand1 * operand2
    if operator == "/":
        return operand1 / operand2
    raise ValueError("未定义运算'{}'".format(operator))


def is_operator(operator: str) -> bool:
    """判断是否为运算符"""
    operator_list = ("+", "-", "*", "/", "(", ")")
    return operator in operator_list


def cut_formula(formula: str) -> list:
    """
    将给定四则运算表达式切分为操作数和操作符的列表

    Dependency: is_operator
    """
    formula = "({})".format(re.sub(r"\s", "", formula))
    token_list = [token for token in re.split(r"([\+\-\*\/\(\)])", formula) if token]
    cutted_formula = []
    negative_flag = False
    for i in range(len(token_list)):
        if token_list[i] == "-":
            if token_list[i - 1] == "(":
                negative_flag = True
                continue
        if not is_operator(token_list[i]):
            cutted_formula.append(("-" if negative_flag else "") + token_list[i])
        else:
            cutted_formula.append(token_list[i])
        negative_flag = False
    return cutted_formula


def make_decision(top_operator: str, now_operator: str) -> int:
    """
    给定栈顶运算符top_operator和当前运算符now_operator,返回决策码

    Return: 1-弹栈并运算, 0-弹栈, -1-压栈
    """
    priority_dict = {
        "+": 1,
        "-": 1,
        "*": 2,
        "/": 2,
        "(": 3,
        ")": 0,
    }
    top_prior = priority_dict.get(top_operator, -1)
    now_prior = priority_dict.get(now_operator, -1)
    if now_prior < 0:
        raise ValueError("未定义运算'{}'".format(now_operator))
    if top_prior == 0:
        return -1
    if top_prior < now_prior:
        return -1
    elif top_prior == 3:
        if now_prior == 0:
            return 0
        else:
            return -1
    else:
        return 1


def stack_calculate(cutted_formula: list) -> float:
    """
    给定切分的表达式cutted_formula,求解表示式值

    Dependency: simple_calculate, is_operator, make_decision
    """
    num_stack = []
    op_stack = []
    for token in cutted_formula:
        if not is_operator(token):
            num_stack.append(float(token))
        else:
            while True:
                if len(op_stack) == 0:
                    op_stack.append(token)
                    break
                decision = make_decision(op_stack[-1], token)
                if decision == -1:
                    op_stack.append(token)
                elif decision == 0:
                    op_stack.pop()
                else:
                    operator = op_stack.pop()
                    operand2 = num_stack.pop()
                    operand1 = num_stack.pop()
                    num_stack.append(simple_calculate(operand1, operand2, operator))
                    continue
                break
    while len(op_stack) > 0:
        operator = op_stack.pop()
        operand2 = num_stack.pop()
        operand1 = num_stack.pop()
        num_stack.append(simple_calculate(operand1, operand2, operator))
    if len(num_stack) != 1:
        raise ValueError("给定表达式非法")
    return num_stack[0]


def expression_calculate(formula: str) -> float:
    """
    计算给定表示式值

    Dependency: cut_formula, stack_calculate
    """
    return stack_calculate(cut_formula(formula))


if __name__ == "__main__":
    print(expression_calculate("-2+3 *5-4. 6/(-3 .2-2)"))
