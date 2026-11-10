"""Preserve explicit units and numeric bounds; no place or preference classifier."""
import copy
import re


def bind_literals(context,query):
    result=copy.deepcopy(context)
    currency_patterns={
        'RUB':r'₽|\bRUB\b|\bруб(?:\.|\b|лей\b|ля\b|ль\b|ли\b)',
        'USD':r'\bUSD\b|\bдоллар(?:ов|а|ы)?\b',
        'EUR':r'€|\bEUR\b|\bевро\b',
    }
    found=[unit for unit,pattern in currency_patterns.items() if re.search(pattern,query,re.I)]
    if len(found)==1 and result['budget']['amount'] is not None:
        result['budget']['currency']=found[0]
    bounds=re.finditer(r'(?:не\s+(?:больше|более|дольше)|максимум|строго(?:\s+до)?|только\s+до)\s*(\d{1,3})\s*мин(?:ут[а-я]*)?\b',query,re.I)
    values=set()
    for match in bounds:
        before=query[max(0,match.start()-45):match.start()]
        after=query[match.end():match.end()+45]
        linked=(re.match(r'\s*(?:пешком|ходьбы|пешего\s+пути|(?:от|до|из)\s+(?:станции\s+)?метро)\b',after,re.I)
                or re.search(r'(?:пешком|(?:от|до|из)\s+метро)\s*[,—–:-]?\s*$',before,re.I))
        if linked and 0<int(match[1])<=120:values.add(int(match[1]))
    if len(values)==1:
        result['proximity']={'minutes':values.pop(),'strict':True}
    return result
