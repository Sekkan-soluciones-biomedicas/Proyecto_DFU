import sqlite3

conn = sqlite3.connect('ResUnet_study.db')
cursor = conn.cursor()

# Ejecuta una consulta para ver las tablas disponibles
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

conn.close()

#------------- Para visualizar los resultados de la base de datos de Optuna desde la terminal: ---------------------

# Correr desde la terminal:
# optuna-dashboard sqlite:///ResUnet_study.db
# y abrir la url que se indica.