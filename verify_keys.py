import os, configparser
from dotenv import load_dotenv
load_dotenv('.env')

class CS(configparser.ConfigParser):
  def optionxform(self, s):
      return s

kd = 'agents/data_agent/DataRetriever_Handbooks/Keys'
for src in ['EPA_AQS', 'OpenTopography', 'OpenWeather', 'US_Census_demography']:
  c = CS()
  c.read(f'{kd}/{src}.keys')
  for k in c['API_Key']:
      raw = c.get('API_Key', k)
      exp = os.path.expandvars(raw)
      ok = exp and exp != raw and not exp.startswith('${')
      masked = (exp[:2] + '***' + exp[-2:]) if (ok and len(exp) > 4) else ('<UNRESOLVED>' if not ok else exp)
      print(('OK   ' if ok else 'FAIL ') + f'{src}/{k}: {masked}')