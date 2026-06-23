from mcp.server import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a:int, b:int) -> int:
    """
    Add to numbers
    """
    return a+b

@mcp.tool()
def Multi(a:int, b:int) -> int:
    """
    Multiplication of Two given integer
    """
    return a * b

if __name__=="__main__":
    mcp.run(transport="stdio")