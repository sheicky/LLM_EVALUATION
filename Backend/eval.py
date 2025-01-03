from google import generativeai as genai 
from openai import OpenAI
from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval,  FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric, ToxicityMetric, SummarizationMetric



import os 
import dotenv 



dotenv.load_dotenv()





# class model : 

#     def __init__(self,query) : 
#         self.query = query 

    
#     def openai_model(self,model_name="llama-3.1-70b-versatile") : 
#         client = OpenAI(
#             model=model_name,
#             api_key=os.getenv("GROQ_API_KEY")
#         )

#         response = client.generate_content(self.query)
#         return response.text

#     def gemini_model(self,model_name="gemini-2.0-flash-exp") : 
#         client = genai.GenerativeModel( 
#             model=model_name,
#             api_key=os.getenv("GEMINI_API_KEY")
#         )

#         response = client.generate_content(self.query)
#         return response.text



class Evaluate : 

    def __init__(self, user_input, llm_output) : 
        self.user_input = user_input
        self.llm_output = llm_output
        self.retrieval_context = None
    

    def deep_eval(self, metric=None):
        test_case = LLMTestCase(input=self.user_input, actual_output=self.llm_output)
        coherence_metric = metric(
            name="Coherence",
            criteria="Coherence - the collective quality of all sentences in the actual output",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        )

        coherence_metric.measure(test_case)
        
        # Formater le résultat pour l'affichage
        result = f"""
        **Score**: {coherence_metric.score}
        
        **Reason**: {coherence_metric.reason}
        """
        return result

    
    def deep_rag_eval(self,metric=None) :  
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = metric(threshold=
        0.5
        )

        metric.measure(test_case)
        print(metric.score)
        print(metric.reason)
        print(metric.is_successful())


    def deep_fine_tunning_eval(self,metric=None) : 
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        context=["..."],
        )
        metric = metric(threshold=
        0.5
        )
        metric.measure(test_case)
        print(metric.score)
        print(metric.is_successful())



    def collect_eval_data(self) : 
        pass 



class Geval(Evaluate) : 
    
    def geval_judge(self,metric=GEval) : 
        return Evaluate.deep_eval(metric)



# class Prometheus(Evaluate) : 
    
#     def prometheus_judge(self,metric=Prometheus) : 
#         return Evaluate.deep_eval(metric)


class RagMetrics(Evaluate) :  


    def faithfullness(self) : 
        return Evaluate.deep_rag_eval(FaithfulnessMetric)

    def answer_relevancy(self) : 
        return Evaluate.deep_rag_eval(AnswerRelevancyMetric)

    def contextual_precision(self) : 
        return Evaluate.deep_rag_eval(ContextualPrecisionMetric)

    def contextual_recall(self) : 
        return Evaluate.deep_rag_eval(ContextualRecallMetric)


    def contextual_relevancy(self) : 
        return Evaluate.deep_rag_eval(ContextualRelevancyMetric)




class FineTuningMetric(Evaluate) :  


    def hallucination(self,metric=HallucinationMetric) :  
        return Evaluate.deep_fine_tunning_eval(metric)

    def toxicity(self) : 
        metric = ToxicityMetric(threshold=
        0.5
        )
        test_case = LLMTestCase(
        input=self.user_input,
        actual_output = self.llm_output
        )

        metric.measure(test_case)
        print(metric.score)


class Summarization(Evaluate) : 

    def summarization(self) : 
        test_case = LLMTestCase(input=self.user_input, actual_output=self.llm_output)
        metric = SummarizationMetric(threshold=
        0.5
        )

        metric.measure(test_case)
        print(metric.score)

