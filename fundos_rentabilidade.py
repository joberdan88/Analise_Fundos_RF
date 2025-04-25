import pandas as pd
import numpy as np

# Mapeamento de meses para números
MESES_MAP = {
    "Janeiro": "1", "Fevereiro": "2", "Março": "3", "Abril": "4",
    "Maio": "5", "Junho": "6", "Julho": "7", "Agosto": "8",
    "Setembro": "9", "Outubro": "10", "Novembro": "11", "Dezembro": "12"
}

# Função para carregar e formatar os dados
def load_data(nome_arquivo):
    try:
        df = pd.read_csv(nome_arquivo, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(nome_arquivo, encoding="latin1")

    # Verifica se o arquivo tem a coluna "Mes"
    if "Mes" in df.columns:
        df["Mes"] = df["Mes"].str.strip().replace(MESES_MAP)
        df["Data"] = pd.to_datetime(df["Ano"].astype(str) + "-" + df["Mes"], format="%Y-%m", errors="coerce")
        df.rename(columns={"Valor_Absoluto(%)": "Rentabilidade", "Valor_CDI(%)": "CDI"}, inplace=True)
    else:
        # Arquivo arrojado: transformar meses em uma única coluna
        df = df.melt(id_vars=["Ano", "No_ano", "Pct_CDI"], var_name="Mes", value_name="Rentabilidade")
        df["Mes"] = df["Mes"].str.strip().replace({
            "Jan": "1", "Fev": "2", "Mar": "3", "Abr": "4", "Mai": "5",
            "Jun": "6", "Jul": "7", "Ago": "8", "Set": "9", "Out": "10",
            "Nov": "11", "Dez": "12"
        })
        df["Data"] = pd.to_datetime(df["Ano"].astype(str) + "-" + df["Mes"], format="%Y-%m", errors="coerce")
        df["CDI"] = df["Pct_CDI"] / 100  # Ajusta CDI para formato decimal

    # Preenche apenas valores numéricos, evitando problemas de tipo na coluna Data
    df.fillna({"Rentabilidade": 0, "CDI": 0}, inplace=True)

    # Remove linhas onde "Data" ainda é inválida antes da ordenação
    df = df[df["Data"].notna()].sort_values("Data").reset_index(drop=True)

    return df[["Data", "Rentabilidade", "CDI"]]

# Função para calcular o Sharpe Ratio
def calculate_sharpe(df, meses):
    df_periodo = df.tail(meses)

    # Remove linhas com NaN dentro do período analisado
    df_valid = df_periodo.dropna(subset=["Rentabilidade", "CDI"])
    if df_valid.empty or len(df_valid) < meses:
        return np.nan

    media_excesso_retorno = (df_periodo["Rentabilidade"] - df_periodo["CDI"]).mean()
    desvio_padrao = df_valid["Rentabilidade"].std()
    if desvio_padrao == 0:
        return np.nan
    return round(media_excesso_retorno / desvio_padrao, 4)

# Execução principal
def main():
    df_arrojado = load_data("rentabilidade_arrojado.csv")
    df_incentivado = load_data("rentabilidade_incentivado.csv")

    for meses in [12, 24, 36]:
        sharpe_arrojado = calculate_sharpe(df_arrojado, meses)
        sharpe_incentivado = calculate_sharpe(df_incentivado, meses)
        print(f"Sharpe Arrojado ({meses} meses): {sharpe_arrojado}")
        print(f"Sharpe Incentivado ({meses} meses): {sharpe_incentivado}")

if __name__ == "__main__":
    main()
