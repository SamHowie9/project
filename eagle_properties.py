import eagleSqlTools as sql
import numpy as np


con = sql.connect("hvq009", password="vzhBQ02t")

query = "SELECT MassType_Star FROM RefL0100N1504_Subhalo"

data = sql.execute_query(con, query)

print(data)

