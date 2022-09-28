import typer
from .etl_baumgartner_suzuki import main as etl_baumgartner_suzuki
from .etl_reizman_suzuki import main as etl_reizman_suzuki


app = typer.Typer()


@app.command()
def baumgartner_suzuki(input_file: str, output_file: str):
    """Extracts the Baumgartner-Suzuki Excel file (input_file) and saves as ORD protobuf (output_file)."""
    etl_baumgartner_suzuki(input_file=input_file, output_file=output_file)


@app.command()
def reizman_suzuki(
    input_file: str,
    output_file: str,
    rxn_sheet_name="Reaction data",
    stock_sheet_name="Stock solutions",
):
    """Extracts the Reizman-Suzuki Excel file (input_file) and saves as ORD protobuf (output_file)."""
    etl_reizman_suzuki(
        input_file=input_file,
        output_file=output_file,
        rxn_sheet_name=rxn_sheet_name,
        stock_sheet_name=stock_sheet_name,
    )


if __name__ == "__main__":
    app()
