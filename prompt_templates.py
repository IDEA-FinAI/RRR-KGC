H2T_TASK_TEMPLATE = "The question is to predict the tail entity [MASK] from the given ({head}, {relation}, [MASK]) by completing the sentence '{sentence}'."
T2H_TASK_TEMPLATE = "The question is to predict the head entity [MASK] from the given ([MASK], {relation}, {tail}) by completing the sentence '{sentence}'."

REASON_TEMPLATE = "Here are some materials for you to refer to. \n{materials}\n\
{task} \
Output all the possible answers you can find in the materials using the format '[answer1, answer2, ..., answerN]' and please start your response with 'The possible answers:'. \
Do not output anything except the possible answers. If you cannot find any answer, please output some possible answers based on your own knowledge."

RANKING_TEMPLATE = "{task} \
The list of candidate answers is [{candidate_list}]. \
{cadidate_list_ctx}\
Sort the list to let the candidate answers which are more possible to be the true answer to the question prior. \
Output the sorted order of candidate answers using the format '[most possible answer, second possible answer, ..., least possible answer]' and please start your response with 'The final order:'. "