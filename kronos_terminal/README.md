# kronos_terminal

Prototype : bougies Binance (API publique) -> Kronos-small -> éventail de trajectoires + lecture du régime.

```bash
git clone https://github.com/shiyu-coder/Kronos ../Kronos
pip install -r ../Kronos/requirements.txt -r requirements.txt
python forecast.py --kronos-dir ../Kronos --symbol BTCUSDT --interval 1h --paths 30
```

Ouvre `forecast.html`. Régime : >=75 % ou <=25 % des trajectoires finissent au-dessus du dernier close = TRENDING, sinon CHOPPY.

Limites : outil exploratoire, pas un signal de trading. Backtest walk-forward avec frais et paper trading avant tout usage réel.
