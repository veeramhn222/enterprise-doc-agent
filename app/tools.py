"""Tools the agents can call."""
import ast
import operator as op

from . import config, vectorstore

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
        ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos, ast.FloorDiv: op.floordiv}


def calculator(expression: str) -> str:
    """Safe arithmetic evaluator (no eval)."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    try:
        expr = expression.replace(",", "").replace("$", "").replace("^", "**").strip()
        val = ev(ast.parse(expr, mode="eval").body)
        return f"{val:,.4f}".rstrip("0").rstrip(".") if isinstance(val, float) else str(val)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as e:
        return f"Error: cannot evaluate '{expression}' ({e})"


def list_documents_tool() -> str:
    docs = vectorstore.list_documents()
    if not docs:
        return "No documents have been uploaded."
    return "\n".join(f"- {d['filename']} ({d['chunks']} chunks, uploaded {d['uploaded_at']})" for d in docs)


def search_documents(query: str, k: int = config.TOP_K) -> list[dict]:
    hits = vectorstore.search(query, k)
    return [h for h in hits if h["distance"] <= config.MAX_DISTANCE]
