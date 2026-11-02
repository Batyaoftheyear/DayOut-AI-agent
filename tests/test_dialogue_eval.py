from eval_runner import dialogue_constraints_match


def test_budget_change_cancellation_and_no_search_are_checked_per_turn():
    case={'budget_checks':[{'turn':1,'expected':{'amount':3000,'scope':'total'}},
                           {'turn':2,'expected':{'amount':2000}}, {'turn':3,'expected':{'amount':None}}],
          'no_tools_turns':[3]}
    turns=[{'completed':True,'tools':['search_places'],'context':{'budget':{'amount':3000,'scope':'total'}}},
           {'completed':True,'tools':['search_places'],'context':{'budget':{'amount':2000}}},
           {'completed':True,'tools':[],'context':{'budget':{'amount':None}}}]
    assert dialogue_constraints_match(turns,case)
    turns[2]['context']['budget']['amount']=2000
    assert not dialogue_constraints_match(turns,case)
    assert not dialogue_constraints_match(turns[:2],case)
