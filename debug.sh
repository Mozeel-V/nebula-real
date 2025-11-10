: << 'LABEL1'
python - <<'PY'
import json
wb = json.load(open("results/sample_probs_before.json"))
wa = json.load(open("results/sample_probs_after.json"))
web = json.load(open("results/window_eval_before/window_eval.json"))["samples"]
wea = json.load(open("results/window_eval_after/window_eval.json"))["samples"]
print("probs_before:",len(wb),"probs_after:",len(wa))
print("we_before samples:",len(web),"we_after samples:",len(wea))
# intersection
common = set(wb.keys()) & set(wa.keys()) & set(web.keys()) & set(wea.keys())
print("common sids:",len(common))
# how many are malware among common
labs = [web[s]["label"] for s in common]
print("malware_in_common:", sum(1 for x in labs if int(x)==1),"benign_in_common:", sum(1 for x in labs if int(x)==0))
PY
LABEL1

python - <<'PY'
import json, numpy as np
b=json.load(open("results/sample_probs_before.json"))
a=json.load(open("results/sample_probs_after.json"))
diffs=[abs(a[k]-b[k]) for k in b if k in a]
print("Mean |prob change|:", np.mean(diffs))
print("Max |prob change|:", np.max(diffs))
print("Samples with >0.01 prob change:", sum(1 for d in diffs if d>0.01))
PY

