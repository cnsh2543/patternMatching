import pytest
import os

from .evaluation import evaluation_function
from .expression_utilities import elementary_functions_names, substitute

# REMARK: If a case is marked with an alternative output, this means that it is difficult in this case to prevent sympy from simplifying for that particular case
elementary_function_test_cases = [
    ("sin", "Bsin(pi)", "0", r"B \sin{\left(\pi \right)}"),
    ("sinc", "Bsinc(0)", "B", r"B 1"),  # r"B \sinc{\left(0 \right)}"
    ("csc", "Bcsc(pi/2)", "B", r"B \csc{\left(\frac{\pi}{2} \right)}"),
    ("cos", "Bcos(pi/2)", "0", r"B \cos{\left(\frac{\pi}{2} \right)}"),
    ("sec", "Bsec(0)", "B", r"B \sec{\left(0 \right)}"),
    ("tan", "Btan(pi/4)", "B", r"B \tan{\left(\frac{\pi}{4} \right)}"),
    ("cot", "Bcot(pi/4)", "B", r"B \cot{\left(\frac{\pi}{4} \right)}"),
    ("asin", "Basin(1)", "B*pi/2", r"B \operatorname{asin}{\left(1 \right)}"),
    ("acsc", "Bacsc(1)", "B*pi/2", r"B \operatorname{acsc}{\left(1 \right)}"),
    ("acos", "Bacos(1)", "0", r"B \operatorname{acos}{\left(1 \right)}"),
    ("asec", "Basec(1)", "0", r"B \operatorname{asec}{\left(1 \right)}"),
    ("atan", "Batan(1)", "B*pi/4", r"B \operatorname{atan}{\left(1 \right)}"),
    ("acot", "Bacot(1)", "B*pi/4", r"B \operatorname{acot}{\left(1 \right)}"),
    ("atan2", "Batan2(1,1)", "B*pi/4", r"\frac{\pi}{4} B"),  # r"B \operatorname{atan2}{\left(1,1 \right)}"
    ("sinh", "Bsinh(x)+Bcosh(x)", "B*exp(x)", r"B \sinh{\left(x \right)} + B \cosh{\left(x \right)}"),
    ("cosh", "Bcosh(1)", "B*cosh(-1)", r"B \cosh{\left(1 \right)}"),
    ("tanh", "2Btanh(x)/(1+tanh(x)^2)", "B*tanh(2*x)", r"\frac{2 B \tanh{\left(x \right)}}{\tanh^{2}{\left(x \right)} + 1}"),  # Ideally this case should print tanh(x)^2 instead of tanh^2(x)
    ("csch", "Bcsch(x)", "B/sinh(x)", r"B \operatorname{csch}{\left(x \right)}"),
    ("sech", "Bsech(x)", "B/cosh(x)", r"B \operatorname{sech}{\left(x \right)}"),
    ("asinh", "Basinh(sinh(1))", "B", r"B \operatorname{asinh}{\left(\sinh{\left(1 \right)} \right)}"),
    ("acosh", "Bacosh(cosh(1))", "B", r"B \operatorname{acosh}{\left(\cosh{\left(1 \right)} \right)}"),
    ("atanh", "Batanh(tanh(1))", "B", r"B \operatorname{atanh}{\left(\tanh{\left(1 \right)} \right)}"),
    ("asech", "Bsech(asech(1))", "B", r"B \operatorname{sech}{\left(\operatorname{asech}{\left(1 \right)} \right)}"),
    ("exp", "Bexp(x)exp(x)", "B*exp(2*x)", r"B e^{x} e^{x}"),
    ("exp2", "a+b*E^2", "a+b*exp(2)", r"a + b e^{2}"),
    ("exp3", "a+b*e^2", "a+b*exp(2)", r"a + b e^{2}"),
    ("ln", "Bexp(ln(10))", "10B", r"B e^{\ln{\left(10 \right)}}"),
    ("log10", "B*10^(log(10,10))", "10B", r"10^{\log_{10}{\left(10 \right)}} B"),
    ("logb", "B*b^(log(a,b))", "aB", r"B b^{\log_{b}{\left(a \right)}}"),
    ("sqrt", "Bsqrt(4)", "2B", r"\sqrt{4} B"),
    ("sign", "Bsign(1)", "B", r"B \operatorname{sign}{\left(1 \right)}"),
    ("abs", "BAbs(-2)", "2B", r"B \left|{-2}\right|"),
    ("Max", "BMax(0,1)", "B", r"B 1"),  # r"B \max{\left(0,1 \right)}"
    ("Min", "BMin(1,2)", "B", "B 1"),  # r"B \min{\left(1,2 \right)}"
    ("arg", "Barg(1)", "0", r"B \arg{\left(1 \right)}"),
    ("ceiling", "Bceiling(0.6)", "B", r"B 1"),  # r"B \left\lceil 0.6 \right\rceil"),
    ("floor", "Bfloor(0.6)", "0", r"B 0"),  # r"B \left\lfloor 0.6 \right\rfloor"),
    ("MECH50001_7.2", "fs/(1-Mcos(theta))", "fs/(1-M*cos(theta))", r"\frac{f s}{1 - M \cos{\left(\theta \right)}}"),
]


def generate_input_variations(response=None, answer=None):
    if response is None or answer is None:
        raise Exception("both response and answer must be specified when generating input variations")
    input_variations = [(response, answer)]
    variation_definitions = [
        lambda x: x.replace('**', '^'),
        lambda x: x.replace('**', '^').replace('*', ' '),
        lambda x: x.replace('**', '^').replace('*', '')
    ]
    for variation in variation_definitions:
        response_variation = variation(response)
        answer_variation = variation(answer)
        if (response_variation != response):
            input_variations.append((response_variation, answer))
        if (answer_variation != answer):
            input_variations.append((response, answer_variation))
        if (response_variation != response) and (answer_variation != answer):
            input_variations.append((response_variation, answer_variation))
    return input_variations


def generate_input_variations_from_elementary_function_aliases(response, answer, params):
    input_variations = [(response, answer)]
    alias_substitutions = []
    for (name, alias) in elementary_functions_names:
        if name in answer or name in response:
            alias_substitutions += [(name, alias)]
    alias_substitutions.sort(key=lambda x: -len(x[0]))
    for k, substitution in enumerate(alias_substitutions):
        for alternative in substitution[1]:
            current_substitutions = [(c, a[0]) for (c, a) in alias_substitutions[0:k]]
            current_substitutions.append((substitution[0], alternative))
            current_substitutions += [(c, a[0]) for (c, a) in alias_substitutions[(k+1):]]
            subs_answer = substitute(answer, current_substitutions)
            subs_response = substitute(response, current_substitutions)
            input_variations += [(subs_response, subs_answer)]
    return input_variations


class TestEvaluationFunction():
    """
    TestCase Class used to test the algorithm.
    ---
    Tests are used here to check that the algorithm written
    is working as it should.

    It's best practise to write these tests first to get a
    kind of 'specification' for how your algorithm should
    work, and you should run these tests before committing
    your code to AWS.

    Read the docs on how to use unittest here:
    https://docs.python.org/3/library/unittest.html

    Use evaluation_function() to check your algorithm works
    as it should.
    """

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="3*x**2 + 3*x +  5",
            answer="2+3+x+2*x + x*x*3"
        )
    )
    def test_simple_polynomial_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="3*longName**2 + 3*longName + 5",
            answer="2+3+longName+2*longName + 3*longName * longName"
        )
    )
    def test_simple_polynomial_with_input_symbols_correct(self, response, answer):
        params = {
            "strict_syntax": False,
            "symbols": {
                "longName": {"aliases": [], "latex": r"\(\mathrm{longName}\)"}
            }
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_simple_polynomial_with_input_symbols_implicit_correct(self):
        response = "abcxyz"
        answer = "abc*xyz"
        params = {
            "strict_syntax": False,
            "symbols": {
                "abc": {"aliases": [], "latex": "\\(abc\\)"},
                "xyz": {"aliases": [], "latex": "\\(xyz\\)"}
            }
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="3*x**2 + 3*x +  5",
            answer="2+3+x+2*x + x*x*3 - x"
        )
    )
    def test_simple_polynomial_incorrect(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is False

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="cos(x)**2 + sin(x)**2 + y",
            answer="y + 1"
        )
    )
    def test_simple_trig_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_expression_with_assumption(self):
        answer = "sqrt(a/b)"
        response = "sqrt(a/b)"
        params = {"strict_syntax": False, "symbol_assumptions": "('a','positive') ('b','positive')"}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="1/( ((x+1)**2) * ( sqrt(1-(x/(x+1))**2) ) )",
            answer="1/((x+1)*(sqrt(2x+1)))"
        )
        +
        generate_input_variations(
            response="1/((x+1)*(sqrt(2x+1)))",
            answer="1/( ((x+1)**2) * ( sqrt(1-(x/(x+1))**2) ) )"
        )
    )
    def test_complicated_expression_correct(self, response, answer):
        params = {"strict_syntax": False, "symbol_assumptions": "('x','positive')"}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    input_variations = []
    fractional_powers_res = ["sqrt(v)/sqrt(g)", "v**(1/2)/g**(1/2)", "v**(0.5)/g**(0.5)"]
    fractional_powers_ans = ["sqrt(v/g)", "(v/g)**(1/2)", "(v/g)**(0.5)"]
    for response in fractional_powers_ans:
        for answer in fractional_powers_ans:
            input_variations += generate_input_variations(response, answer)
    fractional_powers_res = ["v**(1/5)/g**(1/5)", "v**(0.2)/g**(0.2)"]
    fractional_powers_ans = ["(v/g)**(1/5)", "(v/g)**(0.2)"]
    for response in fractional_powers_ans:
        for answer in fractional_powers_ans:
            input_variations += generate_input_variations(response, answer)
    response = "v**(1/n)/g**(1/n)"
    answer = "(v/g)**(1/n)"
    input_variations += generate_input_variations(response, answer)

    @pytest.mark.parametrize("response,answer", input_variations)
    def test_simple_fractional_powers_correct(self, response, answer):
        params = {"strict_syntax": False, "symbol_assumptions": "('g','positive') ('v','positive')"}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_invalid_user_expression(self):
        response = "a*(b+c"
        answer = "a*(b+c)"
        result = evaluation_function(response, answer, {}, include_test_data=True)
        assert "PARSE_ERROR" in result["tags"]

    def test_invalid_author_expression(self):
        response = "3*x"
        answer = "3x"
        e = None
        with pytest.raises(Exception) as e:
            evaluation_function(response, answer, {})
        assert e is not None

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="1+tan(x)**2 + y",
            answer="sec(x)**2 + y"
        )
    )
    def test_recp_trig_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="x/2",
            answer="0.5*x"
        )
    )
    def test_decimals_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="|x|+y",
            answer="Abs(x)+y"
        )
    )
    def test_absolute_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="a|x|+|y|",
            answer="a*Abs(x)+Abs(y)"
        )
        +
        generate_input_variations(
            response="|x|a+|y|",
            answer="a*Abs(x)+Abs(y)"
        )
    )
    def test_absolute_ambiguity(self, response, answer):
        params = {"strict_syntax": False, "elementary_functions": True}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is True
        assert "ABSOLUTE_VALUE_NOTATION_AMBIGUITY" in result["tags"]

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="a*|x+b*|y||",
            answer="a*Abs(x+b*Abs(y))"
        )
    )
    def test_nested_absolute_response(self, response, answer):
        response = "a|x+b|y||"
        answer = "a*Abs(x+b*Abs(y))"
        params = {"strict_syntax": False, "elementary_functions": True}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is True
        assert "ABSOLUTE_VALUE_NOTATION_AMBIGUITY" in result["tags"]

    @pytest.mark.parametrize(
        "response,answer",
        [
            ("|x|+|y|", "Abs(x)+Abs(y)"),
            ("|x|+|y|", "|x|+|y|"),
            ("|x+|y||", "|x+|y||"),
            ("a*|x+b*|y||", "a*|x+b*|y||")
        ]
    )
    def test_many_absolute_response(self, response, answer):
        response = "|x|+|y|"
        answer = "Abs(x)+Abs(y)"
        result = evaluation_function(response, answer, {})
        assert result["is_correct"] is True

    def test_absolute_ambiguity_response(self):
        response = "|a+b|c+d|e+f|"
        answer = "|a+b|*c+d*|e+f|"
        result = evaluation_function(response, answer, {}, include_test_data=True)
        assert "ABSOLUTE_VALUE_NOTATION_AMBIGUITY" in result["tags"]

    def test_absolute_ambiguity_answer(self):
        response = "|a+b|*c+d*|e+f|"
        answer = "|a+b|c+d|e+f|"

        e = None
        with pytest.raises(Exception) as e:
            evaluation_function(response, answer, {})
        assert e is not None
        assert e.value.args[1] == "ABSOLUTE_VALUE_NOTATION_AMBIGUITY"

    def test_clashing_symbols(self):
        params = {}
        response = "beta+gamma+zeta+I+N+O+Q+S+E"
        answer = "E+S+Q+O+N+I+zeta+gamma+beta"
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="pi",
            answer="2*asin(1)"
        )
    )
    def test_special_constants(self, response, answer):
        response = "pi"
        answer = "2*asin(1)"
        params = {"strict_syntax": False, "elementary_functions": True}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(response="I", answer="(-1)**(1/2)")
    )
    def test_complex_numbers(self, response, answer):
        params = {"complexNumbers": True, "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    response = "beta(1,x)"
    answer = "1/x"
    input_variations = generate_input_variations(response, answer)
    response = "gamma(5)"
    answer = "24"
    input_variations += generate_input_variations(response, answer)
    response = "zeta(2)"
    answer = "pi**2/6"
    input_variations += generate_input_variations(response, answer)

    @pytest.mark.parametrize("response,answer", generate_input_variations(response, answer))
    def test_special_functions(self, response, answer):
        params = {"specialFunctions": True, "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="-minus_plus x**2 - plus_minus y**2",
            answer="plus_minus x**2 + minus_plus y**2"
        )
    )
    def test_plus_minus_all_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="- -+ x**2 - +- y**2",
            answer="+- x**2 + -+ y**2"
        )
    )
    def test_plus_minus_replace_symbols_all_correct(self, response, answer):
        params = {"plus_minus": "+-", "minus_plus": "-+", "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="plus_minus x**2 - minus_plus y**2",
            answer="plus_minus x**2 + minus_plus y**2"
        )
    )
    def test_plus_minus_all_incorrect(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is False

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="-minus_plus x**2 - plus_minus y**2",
            answer="plus_minus x**2 + minus_plus y**2"
        )
    )
    def test_plus_minus_all_responses_correct(self, response, answer):
        params = {"multiple_answers_criteria": "all_responses", "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="-x**2 - y**2",
            answer="plus_minus x**2 + minus_plus y**2"
        )
    )
    def test_plus_minus_all_responses_incorrect(self, response, answer):
        params = {"multiple_answers_criteria": "all_responses", "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is False

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="-x**2",
            answer="plus_minus minus_plus x**2"
        )
    )
    def test_plus_minus_all_answers_correct(self, response, answer):
        params = {"multiple_answers_criteria": "all_responses", "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="x**2",
            answer="plus_minus minus_plus x**2"
        )
    )
    def test_plus_minus_all_answers_incorrect(self, response, answer):
        params = {"multiple_answers_criteria": "all_responses", "strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is False

    def test_simplified_in_correct_response(self):
        response = "a*x + b"
        answer = "b + a*x"
        result = evaluation_function(response, answer, {})
        assert result["is_correct"] is True
        assert result["response_simplified"] == "a*x + b"

    def test_simplified_in_wrong_response(self):
        response = "a*x + b"
        answer = "b + a*x + 8"
        result = evaluation_function(response, answer, {})
        assert result["is_correct"] is False
        assert result["response_simplified"] == "a*x + b"

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="2*x**2 = 10*y**2+14",
            answer="x**2-5*y**2-7=0"
        )
    )
    def test_equality_sign_in_answer_and_response_correct(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="2*x**2 = 10*y**2+20",
            answer="x**2-5*y**2-7=0"
        )
    )
    def test_equality_sign_in_answer_and_response_incorrect(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is False

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="2*x**2-10*y**2-14",
            answer="x**2-5*y**2-7=0"
        )
    )
    def test_equality_sign_in_answer_not_response(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is False
        assert "EXPRESSION_NOT_EQUALITY" in result["tags"]

    @pytest.mark.parametrize(
        "response,answer",
        generate_input_variations(
            response="2*x**2 = 10*y**2+14",
            answer="x**2-5*y**2-7"
        )
    )
    def test_equality_sign_in_response_not_answer(self, response, answer):
        params = {"strict_syntax": False}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is False
        assert "EQUALITY_NOT_EXPRESSION" in result["tags"]

    def test_empty_old_format_input_symbols_codes_and_alternatives(self):
        answer = '(1+(gamma-1)/2)((-1)/(gamma-1))'
        response = '(1+(gamma-1)/2)((-1)/(gamma-1))'
        params = {
            'strict_syntax': False,
            'input_symbols': [['gamma', ['']], ['', ['A']], [' ', ['B']], ['C', ['  ']]]
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_empty_input_symbols_codes_and_alternatives(self):
        answer = '(1+(gamma-1)/2)((-1)/(gamma-1))'
        response = '(1+(gamma-1)/2)((-1)/(gamma-1))'
        params = {
            'strict_syntax': False,
            'symbols': {
                'gamma': {'aliases': [''], 'latex': '\\(\\gamma\\)'},
                '': {'aliases': ['A'], 'latex': '\\(A\\)'},
                ' ': {'aliases': ['B'], 'latex': '\\(B\\)'},
                'C': {'aliases': ['  '], 'latex': '\\(C\\)'}
            }
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "description,response,answer,tolerance,tags,outcome",
        [
            (
                "Correct response, tolerance specified with atol",
                "6.73",
                "sqrt(3)+5",
                {"atol": 0.005},
                ["WITHIN_TOLERANCE"],
                True
            ),
            (
                "Incorrect response, tolerance specified with atol",
                "6.7",
                "sqrt(3)+5",
                {"atol": 0.005},
                [],
                False
            ),
            (
                "Correct response, tolerance specified with rtol",
                "6.73",
                "sqrt(3)+5",
                {"rtol": 0.0005},
                ["WITHIN_TOLERANCE"],
                True
            ),
            (
                "Incorrect response, tolerance specified with rtol",
                "6.7",
                "sqrt(3)+5",
                {"rtol": 0.0005},
                [],
                False
            ),
            (
                "Response is not constant, tolerance specified with atol",
                "6.7+x",
                "sqrt(3)+5",
                {"atol": 0.005},
                ["NOT_NUMERICAL"],
                False
            ),
            (
                "Answer is not constant, tolerance specified with atol",
                "6.73",
                "sqrt(3)+x",
                {"atol": 0.005},
                ["NOT_NUMERICAL"],
                False
            ),
            (
                "Response is not constant, tolerance specified with rtol",
                "6.7+x",
                "sqrt(3)+5",
                {"rtol": 0.0005},
                ["NOT_NUMERICAL"],
                False
            ),
            (
                "Answer is not constant, tolerance specified with rtol",
                "6.73",
                "sqrt(3)+x",
                {"rtol": 0.0005},
                ["NOT_NUMERICAL"],
                False
            ),
        ]
    )
    def test_numerical_comparison_problem(self, description, response, answer, tolerance, tags, outcome):
        params = {"numerical": True}
        params.update(tolerance)
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is outcome
        for tag in tags:
            tag in result["tags"]

    def test_warning_inappropriate_symbol(self):
        answer = 'factorial(2**4)'
        response = '2^4!'
        params = {'strict_syntax': True}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is False
        assert "NOTATION_WARNING_EXPONENT" in result["tags"]
        assert "NOTATION_WARNING_FACTORIAL" in result["tags"]

    @pytest.mark.parametrize(
        "response,answer",
        [
            (
                '0,5',
                '0.5'
            ),
            (
                '(0,002*6800*v)/1,2',
                '(0,002*6800*v)/1.2'
            ),
            (
                '-∞',
                '-inf'
            ),
            (
                'x.y',
                'x*y'
            ),
        ]
    )
    def test_error_inappropriate_symbol(self, response, answer):
        params = {'strict_syntax': True}
        result = evaluation_function(response, answer, params, include_test_data=True)
        assert result["is_correct"] is False
        assert "PARSE_ERROR" in result["tags"]

    @pytest.mark.parametrize(
        "description,response",
        [
            (
                "Empty response",
                "",
            ),
            (
                "Whitespace response",
                "  \t\n",
            ),
        ]
    )
    def test_empty_response(self, description, response):
        answer = "5*x"
        result = evaluation_function(response, answer, {}, include_test_data=True)
        assert "NO_RESPONSE" in result["tags"]

    @pytest.mark.parametrize(
        "description,answer",
        [
            (
                'Empty answer',
                ''
            ),
            (
                'Whitespace answer',
                '  \t\n'
            ),
        ]
    )
    def test_empty_answer(self, description, answer):
        response = "5*x"
        e = None
        with pytest.raises(Exception) as e:
            evaluation_function(response, answer, {})
        assert e is not None
        assert e.value.args[0] == "No answer was given."

    @pytest.mark.parametrize(
        "description,response,answer,outcome",
        [
            (
                "With `fx` in response",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                True
            ),
            (
                "Without `-` in response",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                "A*exp(x/b)*sin(y/b)+fx+C",
                False
            ),
            (
                "With `f(x)` in response",
                "A*exp(x/b)*sin(y/b)+f(x)+C",
                "-A*exp(x/b)*sin(y/b)+f(x)+C",
                False
            ),
        ]
    )
    def test_slow_response(self, description, response, answer, outcome):
        params = {
            "strict_syntax": False,
            "symbols": {
                "fx": {"aliases": ["f", "f_x", "fofx"], "latex": "\\(f(x)\\)"},
                "C": {"aliases": ["c", "k", "K"], "latex": "\\(C\\)"},
                "A": {"aliases": ["a"], "latex": "\\(A\\)"},
                "B": {"aliases": ["b"], "latex": "\\(B\\)"},
                "x": {"aliases": ["X"], "latex": "\\(x\\)"},
                "y": {"aliases": ["Y"], "latex": "\\(y\\)"},
            }
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is outcome

    @pytest.mark.parametrize(
        "description,response,answer,outcome",
        [
            (
                "With `fx` in response",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                True
            ),
            (
                "Without `-` in response",
                "-A*exp(x/b)*sin(y/b)+fx+C",
                "A*exp(x/b)*sin(y/b)+fx+C",
                False
            ),
            (
                "With `f(x)` in response",
                "A*exp(x/b)*sin(y/b)+f(x)+C",
                "-A*exp(x/b)*sin(y/b)+f(x)+C",
                False
            ),
        ]
    )
    def test_slow_response_old_format_input_symbols(self, description, response, answer, outcome):
        params = {
            "strict_syntax": False,
            "input_symbols": [
                ["fx", ["f", "f_x", "fofx"]],
                ["C", ["c", "k", "K"]],
                ["A", ["a"]],
                ["B", ["b"]],
                ["x", ["X"]],
                ["y", ["Y"]]
            ]
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is outcome

    def test_pi_with_rtol(self):
        answer = "pi"
        response = "3.14"
        params = {
            "strict_syntax": False,
            "rtol": 0.05,
            "symbols": {
                "pi": {"aliases": ["Pi", "PI", "π"], "latex": "\\(\\pi\\)"},
            }
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "response,outcome",
        [
            (
                res,
                False
            ) for res in [
                "-(sin(xy)y+(e^y))/(x(e^y+sin(xy)x))"
                "sin(xy)y",
                "sin(xy)x",
                "x(e^y+sin(xy)x)",
                "e^y+sin(xy)x"
            ]
        ]
    )
    def test_PHYS40002_2_2_b(self, response, outcome):
        params = {"strict_syntax": False}
        answer = "-(y*sin(x*y) + e^(y)) / (x*(e^(y) + sin(x*y)))"
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is outcome

    @pytest.mark.parametrize(
        "response,outcome",
        [
            (
                res,
                True
            ) for res in [
                "6*cos(5*x+1)-90*x*sin(5*x+1)-225*x**2*cos(5*x+1)+125*x**3*sin(5*x+1)",
                "6cos(5x+1)-90x*sin(5x+1)-225x^2cos(5x+1)+125x^3sin(5x+1)",
            ]
        ]+[
            (
                res,
                False
            ) for res in [
                "-90xsin(5x+1)",
                "6cos(5x+1)-90xsin(5x+1)-225x^2cos(5x+1)+125x^3sin(5x+1)",
                "(125x^3)*(cos(5x+1))-(225x^2)*(cos(5x+1))-(90x)*(sin(5x+1))+6cos(5x+1)",
            ]
        ]
    )
    def test_PHYS40002_2_6_a(self, response, outcome):
        params = {"strict_syntax": False}
        answer = "6*cos(5*x+1)-90*x*sin(5*x+1)-225*x**2*cos(5*x+1)+125*x**3*sin(5*x+1)"
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is outcome

    @pytest.mark.parametrize(
        "description,response,answer",
        sum(
            [
                [
                    (case[0],)+var
                    for var in generate_input_variations_from_elementary_function_aliases(
                        case[1],
                        case[2],
                        {"strict_syntax": False, "elementary_functions": True}
                    )
                ]
                for case in elementary_function_test_cases
            ],
            []
        )
    )
    def test_elementary_functions(self, description, response, answer):
        params = {"strict_syntax": False, "elementary_functions": True}
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    @pytest.mark.parametrize(
        "res,ans,convention",
        [
            ("1/ab", "1/(ab)", "implicit_higher_precedence"),
            ("1/ab", "1/a*b", "equal_precedence"),
            ("1/ab", "(1/a)*b", "equal_precedence"),
        ]
    )
    def test_implicit_multiplication_convention(self, res, ans, convention):
        params = {"strict_syntax": False, "convention": convention}
        result = evaluation_function(res, ans, params)
        assert result["is_correct"] is True

    def test_eval_function_can_handle_latex_input(self):
        response = r"\sin x + x^{7}"
        answer = "sin(x)+x**7"
        params = {
            "strict_syntax": False,
            "is_latex": True
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_replace_lambda(self):
        params = {
            "strict_syntax": False,
            "elementary_functions": True,
        }
        # Since 'lambda' is a reserved keyword in python this case would
        # cause a parsing error unless 'lambda' is replaced appropriately
        response = "lambda"
        answer = "lambda"
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

    def test_exclamation_mark_for_factorial(self):
        response = "3!"
        answer = "factorial(3)"
        params = {
            "strict_syntax": False,
            "elementary_functions": True,
        }
        result = evaluation_function(response, answer, params)
        assert result["is_correct"] is True

if __name__ == "__main__":
    pytest.main(['-xsk not slow', "--tb=line", os.path.abspath(__file__)])
