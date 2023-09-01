import openai
import json
from utils import stem, pos_tag_method, tokenize_process
import http
from bert_score import score
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_trf")

openai.api_key = "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"
serper_key = "45a045c883ae768b9b94bb5dc1577fcda3886590"

class fact_gen():
    def __init__(self, model="gpt-3.5-turbo", output_file=None):
        self.model_name = model
        if output_file:
            self.output_file = open(output_file, "a", buffering=1)
        else:
            self.output_file = None

    def generate(self, prompt, temp=1, min_tokens=512, sent_max_tokens=128):
        result_str = ""
        chat_history = [     
            {"role": "system", "content": "You are a helpful assistant designed to help me produce accurate and factual texts based on the given prompt. Please generate only one sentence at a time."},
            {"role": "user", "content": prompt}
        ]
        overall_extrinsic = []
        while len(result_str) < min_tokens:
            init_new_sent = self.sent_generate(chat_history, temp, sent_max_tokens)
            if self.output_file:
                print(init_new_sent, file=self.output_file)
            extrinsic_list, intrinsic_list = self.validate(init_new_sent)
            overall_extrinsic.extend(extrinsic_list)
            if len(intrinsic_list) == 0:
                chat_history.extend(
                    [{"role": "assistant", "content": init_new_sent}]
                    #{"role": "user", "content": "Continue."}]
                )
                result_str += init_new_sent
            else:
                # Intrinsic Hallucination Caught
                revised_sent = self.revise(init_new_sent, intrinsic_list, max_tokens=sent_max_tokens, temp=0, n=1)

                chat_history.extend(
                    [{"role": "assistant", "content": revised_sent}]
                    #{"role": "user", "content": "Continue."}]
                )

                result_str += revised_sent
        if self.output_file:
            print(f"chat history: \n{chat_history}", file=self.output_file)
        if len(overall_extrinsic) > 0:
            result_str = self.user_warning(result_str, overall_extrinsic)
        
        return result_str

    def user_warning(self, return_string, extrinsic_list):
        return_string += "\n\nWARNING: The following claims have been generated that might be not objectively correct:\n\n"
            
        for item in extrinsic_list:
            return_string += "Entity: {}\nClaim: {}\n\n".format(item["entity"], item["claim"])
        
        return_string += "Please note that as an AI model, I cannot guarantee the correctness of these facts. \nIt's essential not to rely solely on these without proper verification from trusted sources."
        
        return return_string

    def revise(self, sent, intrinsic_list, max_tokens=64, temp=0, n=1):
        prompt = f"For the original text, please revise each wrong entity mentioned below, along with its associated wrong claim, based on the provided evidence.\n\nOriginal text: {sent}\n\n"
        for ele in intrinsic_list:
            ent = ele["entity"]
            claim = ele["claim"]
            evidence = ele["evidence"]
            prompt += f"Wrong entity: {ent}\nWrong claim:{claim}\nEvidence:{evidence}\n\n"

        prompt += "Revised text: "
        revised_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant. Based on the provided evidence, you will rewrite the text to correct the facts."},
                    {"role": "user", "content": prompt}
                ],
            max_tokens=max_tokens,
            temperature=temp,
            n=n,
           # stop="."
        )
        
        revised = revised_response["choices"][0]["message"]["content"]

        if "\n" in revised:
           revised = revised[:revised.index("\n")]
        return revised
    

    def sent_generate(self, history, temp=1, max_tokens=256):
        """
        Just generate one sentence given the prompt.
        """
        sent_gen = openai.ChatCompletion.create(
            model=self.model_name,
            messages=history,
            max_tokens=max_tokens,
            n=1,
            temperature=temp,
            #stop="."
        )
        
        sent_gen_content = sent_gen["choices"][0]["message"]["content"]
        return sent_gen_content
    
    def validate(self, sent):
        entity_set = self.extract_entity(sent)
        if self.output_file:
            print(f"entity set:\n{entity_set}", file=self.output_file)
        
        fact_dic = self.extract_facts(sent, entity_set, temp=1, n=5)
        if self.output_file:
            print(f"fact_dict:\n{fact_dic}", file=self.output_file)
       
        evidence_dict = self.group_search(fact_dic)
        if self.output_file:
            print(f"evidence dic:\n{evidence_dict}", file=self.output_file)
        
        extrinsic_list = []
        intrinsic_list = []

        for ele in evidence_dict["atomic_fact"]:
            entity = ele["entity"]    
            claim = ele["claim"]                
            evidence_list = ele["evidence"]    
            
            # could incorporate more pieces of evidence
            evidence = evidence_list[0]
            if not self.is_relevant(claim, evidence, entity):

                # keyword search again
                doc = nlp(claim)
                # Find the subject by looking for the word whose dependency relation is "nsubj" (nominal subject)
                subject = [token.text for token in doc if "subj" in token.dep_][0]

                keyword_q = tokenize_process(entity)

                keyword_q.append(subject)

                keyword_q_str = ''
                for word in keyword_q:
                    keyword_q_str += f'"{word}";'
                
                keyword_q_str = keyword_q_str[:-1]

                evidence_keyword = self.single_search(keyword_q_str)["evidence"]

                if not self.is_relevant(claim, evidence_keyword, entity):
                    extrinsic_list.append({
                    "entity": entity,
                    "claim": claim,
                    "evidence": evidence_keyword
                })
                else:
                    if not self.fact_check(claim, evidence, entity):
                        intrinsic_list.append(
                            {
                            "entity": entity,
                            "claim": claim,
                            "evidence": evidence
                            }
                        )
        if self.output_file:
            print("ex", extrinsic_list, file=self.output_file)
            print("in", intrinsic_list, file=self.output_file)

        return extrinsic_list, intrinsic_list


    def fact_check(self, claim, evidence, entity, temp=0, n=1):

        with open("demos/demon_fact-check.txt", "r") as f:
            response_template = f.read()
            #response_template += "Evidence: {}\nClaim: {}\nBased on the evidence and claim above, the evidence is "
            response_template += "Evidence: {}\nClaim: {}\nOnly based on the evidence above, the claim is "

        filled_prompt = response_template.format(evidence, claim)

        factcheck_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    #{"role": "system", "content": "You are a fact-checking assistant. Based on the provided evidence, you will first determine if it is related and useful to support or contradict the claim."},
                    {"role": "system", "content": "You are a fact-checking assistant. Based on the provided evidence, you will check if the claim is true or false given the provided evidence."},

                    {"role": "user", "content": filled_prompt}
                ],
            max_tokens=4,
            temperature=temp,
            n=n
            )
        
        factcheck_result = factcheck_response["choices"][0]["message"]["content"].lower()
        if factcheck_result[-1] == ".":
            factcheck_result = factcheck_result[:-1]
        if factcheck_result == "true":
            return True
        elif factcheck_result == "false":
            return False
        return factcheck_result

    def single_atmoic_gpt3(self, context, entity, temp=1, n=3):
        with open("demos/demo_atomic_replace_single.txt", "r") as f:
            response_template = f.read()
            #response_template += "Evidence: {}\nClaim: {}\nBased on the evidence and claim above, the evidence is "
            response_template += "Context: {}\n\Standalone Fact: "

        filled_prompt = response_template.format(context, entity)

        atomic_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    #{"role": "system", "content": "You are a fact-checking assistant. Based on the provided evidence, you will first determine if it is related and useful to support or contradict the claim."},
                    {"role": "system", "content": "You are a assistant to help me complete a atomic fact for the factual key entity."},

                    {"role": "user", "content": filled_prompt}
                ],
            max_tokens=128,
            n=n,
            temperature=temp
            )
        atomic_list = list()
        for i in range(len(atomic_response["choices"])):
            atomic_result = atomic_response["choices"][i]["message"]["content"]
            atomic_list.append(atomic_result)
        return atomic_list


    def extract_facts(self, sent, entity_set, temp=0, n=5):
        dic = {
            "text": sent,
            "entities": []
        }
        for entity in entity_set:
            replaced_text = sent.replace(entity, "<entity>")
            res = self.single_atmoic_gpt3(replaced_text, entity, temp=temp, n=n)

            filtered_set = set(res)

            #accepted_facts = process_facts(res, entity, subject)
            accepted_fact = max(filtered_set, key = res.count).replace("<entity>", entity)
            
            dic["entities"].append({
                entity: accepted_fact
            })

        return dic
    
    def single_search(self, query):
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
        "q": query,
        "autocorrect": False,
        "hl": "en",
        "type": "search",
        })
        headers = {
        'X-API-KEY': serper_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = res.read()
        decoded = data.decode("utf-8")
        dic = json.loads(decoded)
        
        knowledge_dict = {}
        evidence_list = []
        if 'answerBox' in dic.keys():
            if "answer" in dic["answerBox"]:
                evidence_list.append(dic["answerBox"]["answer"])
            elif "snippet" in dic["answerBox"]:
                evidence_list.append(dic["answerBox"]["snippet"])

        elif "knowledgeGraph" in dic.keys():
            evidence_list.append(dic["knowledgeGraph"]["description"])
        
        elif 'organic' in dic.keys():
            if len(dic["organic"]) > 0:
                evidence_list.append(dic["organic"][0]["snippet"])

        # could change the number of evidence used
        knowledge_dict = {}
        knowledge_dict["evidence"] = evidence_list
        knowledge_dict["claim"] = dic["searchParameters"]["q"]

        return knowledge_dict
    
    def is_relevant(self, query, evidence, entity, threshold=0.25):

        q_list = set(pos_tag_method(query.lower().replace(entity.lower(),"")))
        evid_list = set(pos_tag_method(evidence.lower().replace(entity.lower(),"")))
 
        precision = len(q_list & evid_list) / len(q_list) if len(q_list) != 0 else 0
        #return True

        return precision >= threshold or entity.lower() in evidence.lower()
    
    def group_search(self, fact_dic, quote_subject=True):
        ret_dic = {
            "text": fact_dic["text"],
            "atomic_fact" : []
        }
        for i in range(len(fact_dic["entities"])):
            entity, atomic_fact = list(fact_dic["entities"][i].items())[0]

            if quote_subject:
                doc = nlp(atomic_fact)
                # Find the subject by looking for the word whose dependency relation is "nsubj" (nominal subject)
                subject = [token.text for token in doc if "subj" in token.dep_][0]

                atomic_fact.replace(subject, f'"{subject}"')
            
            retrived_dict = self.single_search(atomic_fact)
            ret_dic["atomic_fact"].append({
                "entity": entity,
                "claim": atomic_fact,
                "evidence": retrived_dict["evidence"]
            })
        return ret_dic


    def extract_entity(self, sent, temp=0, n=1):
        with open("demos/demo_extract.txt", "r") as f:
            response_template = f.read()
            #response_template += "Evidence: {}\nClaim: {}\nBased on the evidence and claim above, the evidence is "
            response_template += "Sentence: {}\nEntities: "

        filled_prompt = response_template.format(sent)

        extract_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    #{"role": "system", "content": "You are a fact-checking assistant. Based on the provided evidence, you will first determine if it is related and useful to support or contradict the claim."},
                    {"role": "system", "content": "You are a assistant to help me extract factual key phrase in a sentence. "},

                    {"role": "user", "content": filled_prompt}
                ],
            max_tokens=256,
            n=n,
            temperature=temp
            )
        extract_set = set()
        for i in range(len(extract_response["choices"])):
            extract_result = extract_response["choices"][i]["message"]["content"]
            extract_set.update(extract_result.split("; "))
        extract_set = [ent for ent in extract_set if not sent[:len(ent)] == ent and ent in sent]

        # 1. Remove entities that are substrings of another entity
        extract_set = [ent for ent in extract_set if not any(ent in ref and ent != ref for ref in extract_set)]

        # 2. only one entity remains for multiple entities with the same stemmed strings
        seen = set()
        filtered_union = []

        for ent in extract_set:
            stemmed_ent = stem(ent)
            if stemmed_ent not in seen:
                seen.add(stemmed_ent)
                filtered_union.append(ent)
        extract_set = filtered_union

        return extract_set
