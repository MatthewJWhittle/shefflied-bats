# Command Line Interface for species-sdm
import typer

app = typer.Typer()

@app.command()
def hello(name: str = "World"):
    print(f"Hello {name} from species-sdm CLI!")

if __name__ == "__main__":
    app() 