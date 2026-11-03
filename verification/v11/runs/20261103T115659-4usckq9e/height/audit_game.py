"""Read-only independent count/mask audit for the nonzero game test classes."""
import json,zipfile,math,struct
from collections import Counter,defaultdict
from fractions import Fraction as F
import independent as m
with zipfile.ZipFile(m.ROOT/'out6/game.zip') as z:
    g=json.loads(z.read('game.json'));qs=json.loads(z.read('questions.json'))
qa=qs['A'];qb=qs['B'];assert len(qa)==len(qb)==g['questions_per_party']==169460
assert qa[g['old_questions']:]==qb[g['old_questions']:]
sync=set(g['synchrony']);binds=Counter(tuple(x) for x in g['bindings'])
orth={tuple(x) for x in g['orthogonality']}
oldid={q['name']:q['id'] for q in qa[:g['old_questions']]}
comm=defaultdict(list);slots=Counter();potential_pairs=Counter()
for q in qa[g['old_questions']:]:
    if q['kind']=='commute_old':
        comm[q['variable']].append(q)
        assert q['id'] in sync
        # One mismatch binding to the old binary question, in both orientations.
        assert binds[q['id'],'12',q['p'],'2']==1
        potential_pairs[q['id'],q['id']]+=1
        potential_pairs[q['id'],q['p']]+=1;potential_pairs[q['p'],q['id']]+=1
for name in g['core_variables']:
    assert len(comm[name])==16
    assert {(q['axis'],q['letter']) for q in comm[name]}=={(i,a) for i in range(8) for a in ('X','Z')}
    assert oldid[name] in sync;potential_pairs[oldid[name],oldid[name]]+=1
for ri,row in enumerate(g['rows']):
    d=row['d'];assert len(row['slots'])==d and row['slots'][0]==[None,1]
    cols=qa[row['question']];assert cols['kind']=='row_columns' and cols['row']==ri
    assert cols['id'] in sync;potential_pairs[cols['id'],cols['id']]+=1
    coeff=Counter()
    for name,sign in row['slots']:
        if name is None:coeff['I']+=sign
        else:coeff[name]+=2*sign;coeff['I']-=sign
    original=m.CORE['projection_linear_relations'][ri]
    assert original['name']==row['name']
    assert {k:F(v) for k,v in coeff.items() if v}=={k:2*row['den']*F(v) for k,v in original['terms'].items() if F(v)}
    for j,ji in enumerate(row['joint_questions'],1):
        name,sign=row['slots'][j];q=qa[ji]
        assert (q['row'],q['slot'],q['variable'],q['sign'])==(ri,j,name,sign)
        assert q['answers']==[[e,k] for e in (0,1) for k in (0,1,2)]
        assert ji in sync
        assert binds[ji,'56',oldid[name] if name is not None else 0,'2' if name is not None else '1']==1
        potential_pairs[ji,ji]+=1;potential_pairs[cols['id'],ji]+=1
        if name is not None:
            slots[name]+=1;potential_pairs[ji,oldid[name]]+=1;potential_pairs[oldid[name],ji]+=1
        assert (cols['id'],ji,j,sign) in orth
    assert (cols['id'],g['globals']['X'],0,d-1) in orth
    potential_pairs[cols['id'],g['globals']['X']]+=1
assert max(potential_pairs.values())==1
external=[]
for key,num in struct.iter_unpack('<Qq',m.RAW):
    q,a=divmod(key>>32,16);r,b=divmod(key&0xffffffff,16)
    if qa[q]['kind'] in ('fms','refined') or qb[r]['kind'] in ('fms','refined'):
        assert num<0;external.append([q,a,r,b])
assert external==g['old_forbidden']
result={'Q':len(qa),'variables_with_16_joint_contexts_and_32_oriented_bindings':len(comm),'rows_checked':len(g['rows']),'slot_contexts_checked':sum(len(r['joint_questions']) for r in g['rows']),'retained_old_events_checked':len(external),'nonzero_rule_classes_have_disjoint_ordered_question_pairs':True}
print(json.dumps(result,indent=2));(m.OUT/'game_audit.json').write_text(json.dumps(result,indent=2))
