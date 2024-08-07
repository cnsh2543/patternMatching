import os
import pytest

from .preview import Params, extract_latex, preview_function
from .evaluation_tests import elementary_function_test_cases


class TestPreviewFunction():
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

    Use preview_function() to check your algorithm works
    as it should.
    """

    def test_empty_latex_expression(self):
        response = ""
        params = Params(is_latex=True)
        result = preview_function(response, params)
        assert "preview" in result.keys()

        preview = result["preview"]
        assert preview["latex"] == ""

    def test_empty_sympy_expression(self):
        response = ""
        params = Params(is_latex=False)
        result = preview_function(response, params)
        assert "preview" in result.keys()

        preview = result["preview"]
        assert preview["sympy"] == ""

    def test_latex_and_sympy_are_returned(self):
        response = "x+1"
        params = Params(is_latex=True)
        result = preview_function(response, params)
        assert "preview" in result.keys()

        preview = result["preview"]
        assert "latex" in preview.keys()
        assert "sympy" in preview.keys()

        response = "x+1"
        params = Params(is_latex=False)
        result = preview_function(response, params)
        assert "preview" in result

        preview = result["preview"]
        assert "latex" in preview
        assert "sympy" in preview

    def test_doesnt_simplify_latex_by_default(self):
        response = "\\frac{x + x^2 + x}{x}"
        params = Params(is_latex=True)
        result = preview_function(response, params)
        preview = result["preview"]

        assert preview.get("sympy") == "(x**2 + x + x)/x"

    def test_doesnt_simplify_sympy_by_default(self):
        response = "(x + x**2 + x)/x"
        params = Params(is_latex=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("latex") == "\\frac{x^{2} + x + x}{x}"

    def test_simplifies_latex_on_param(self):
        response = "\\frac{x + x^2 + x}{x}"
        params = Params(is_latex=True, simplify=True)
        result = preview_function(response, params)
        preview = result["preview"]

        assert preview.get("sympy") == "x + 2"

    def test_simplifies_sympy_on_param(self):
        response = "(x + x**2 + x)/x"
        params = Params(is_latex=False, simplify=True)
        result = preview_function(response, params)
        preview = result["preview"]

        assert preview.get("latex") == "x + 2"

    def test_sympy_handles_implicit_multiplication(self):
        response = "sin(x) + cos(2x) - 3x**2"
        params = Params(is_latex=False, strict_syntax=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("latex") == "- 3 x^{2} + \\sin{\\left(x \\right)} + \\cos{\\left(2 x \\right)}"

    def test_latex_with_equality_symbol(self):
        response = "\\frac{x + x^2 + x}{x} = y"
        params = Params(is_latex=True, simplify=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("sympy") == "Eq((x**2 + x + x)/x, y)"

    def test_sympy_with_equality_symbol(self):
        response = "Eq((x + x**2 + x)/x, 1)"
        params = Params(is_latex=False, simplify=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("latex") == "\\frac{x^{2} + x + x}{x} = 1"

    def test_latex_conversion_preserves_default_symbols(self):
        response = "\\mu + x + 1"
        params = Params(is_latex=True, simplify=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("sympy") in "mu + x + 1"

    def test_sympy_conversion_preserves_default_symbols(self):
        response = "mu + x + 1"
        params = Params(is_latex=False, simplify=False)
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("latex") == "\\mu + x + 1"

    def test_sympy_conversion_preserves_optional_symbols(self):
        response = "m_table + test + x + 1"
        params = Params(
            is_latex=False,
            simplify=False,
            symbols={
                "m_table": {"latex": "m_{\\text{table}}", "aliases": []},
                "test": {
                    "latex": "\\text{hello}_\\text{world}",
                    "aliases": [],
                },
            },
        )
        result = preview_function(response, params)
        preview = result["preview"]
        assert preview.get("latex") == "m_{\\text{table}} + \\text{hello}_\\text{world} + x + 1"

    def test_invalid_latex_returns_error(self):
        response = "\frac{ m_{ \\text{table} } + x + 1 }{x"
        params = Params(
            is_latex=True,
            simplify=False,
            symbols={"m_table": {"latex": "m_{\\text{table}}", "aliases": []}},
        )

        with pytest.raises(ValueError):
            preview_function(response, params)

    def test_invalid_sympy_returns_error(self):
        response = "x + x***2 - 3 / x 4"
        params = Params(simplify=False, is_latex=False)

        with pytest.raises(ValueError):
            preview_function(response, params)

    def test_extract_latex_in_delimiters(self):
        parentheses = r"\( x + 1 \)"
        dollars = r"$ x ** 2 + 1 $"
        double_dollars = r"$$ \sin x + \tan x $$"

        assert extract_latex(parentheses) == " x + 1 "
        assert extract_latex(dollars) == " x ** 2 + 1 "
        assert extract_latex(double_dollars) == r" \sin x + \tan x "

    def test_extract_latex_in_delimiters_and_text(self):
        parentheses = r"hello \( x + 1 \) world."
        dollars = r"hello $ x ** 2 + 1 $ world."

        assert extract_latex(parentheses) == " x + 1 "
        assert extract_latex(dollars) == " x ** 2 + 1 "

    def test_extract_latex_no_delimiters(self):
        test = r"'\sin x + \left ( \text{hello world} \right ) + \cos x"
        assert extract_latex(test) == test

    def test_extract_latex_multiple_expressions(self):
        parentheses = r"hello \( x + 1 \) world. \( \sin x + \cos x \) yes."
        dollars = r"hello $ x ** 2 + 1 $ world. \( \sigma \times \alpha \) no."
        mixture = r"hello $ x ** 2 - 1 $ world. $ \sigma \times \alpha $ !."

        assert extract_latex(parentheses) == " x + 1 "
        assert extract_latex(dollars) == " x ** 2 + 1 "
        assert extract_latex(mixture) == " x ** 2 - 1 "

    def test_elementary_functions_preview(self):
        params = {"strict_syntax": False, "elementary_functions": True}
        for case in elementary_function_test_cases:
            response = case[1]
            result = preview_function(response, params)
            assert result["preview"]["latex"] == case[3]

    def test_implicit_multiplication_convention_implicit_higher_precedence(self):
        response = "1/ab"
        latex = r"1 \cdot \frac{1}{a b}"  # REMARK: Here it would be preferable to not have the '1 \cdot ' at the start
        params = {"strict_syntax": False, "convention": "implicit_higher_precedence"}
        result = preview_function(response, params)
        assert result["preview"]["latex"] == latex

    def test_implicit_multiplication_convention_equal_precedence(self):
        latex = r"1 \cdot \frac{1}{a} b"  # REMARK: Here it would be preferable to not have the '1 \cdot ' at the start
        params = {"strict_syntax": False, "convention": "equal_precedence"}
        response_a = "1/ab"
        result_a = preview_function(response_a, params)
        assert result_a["preview"]["latex"] == latex
        response_b = "1/a*b"
        result_b = preview_function(response_b, params)
        assert result_b["preview"]["latex"] == latex


if __name__ == "__main__":
    pytest.main(['-sk not slow', "--tb=line", os.path.abspath(__file__)])
