from google import generativeai as genai 
from openai import OpenAI
from deepeval.test_case import LLMTestCase, LLMTestCaseParams 
from deepeval.metrics import GEval,  FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric, HallucinationMetric, ToxicityMetric, SummarizationMetric
#from langchain_openai import ChatOpenAI



import os 
import dotenv 



dotenv.load_dotenv()







class Evaluate : 

    def __init__(self, user_input, llm_output) : 
        self.user_input = user_input
        self.llm_output = llm_output
        self.retrieval_context = None
    

    def deep_eval(self, metric=None):
        test_case = LLMTestCase(input=self.user_input, actual_output=self.llm_output)
        coherence_metric = GEval(
            name="Coherence",
            criteria="Coherence - the collective quality of all sentences in the actual output",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model="gpt-4"
        )

        try:
            coherence_metric.measure(test_case)
            result = f"""
            **Score**: {coherence_metric.score}
            
            **Reason**: {coherence_metric.reason}
            """
            return result
        except Exception as e:
            return f"Evaluation failed: {str(e)}"

    
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
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = FaithfulnessMetric(threshold=0.5,model="gpt-4-0613")
        

        metric.measure(test_case)
        return {
            "score": metric.score,
            "reason": metric.reason,
            "success": metric.is_successful()
        }

    def answer_relevancy(self) : 
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = AnswerRelevancyMetric(threshold=0.5,model="gpt-4-0613")
        metric.measure(test_case)
        return {"score": metric.score}

    def contextual_precision(self) : 
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = ContextualPrecisionMetric(threshold=0.5,model="gpt-4")
        metric.measure(test_case)
        return {"score": metric.score}

    def contextual_recall(self) : 
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = ContextualRecallMetric(threshold=0.5,model="gpt-4")
        metric.measure(test_case)
        return {"score": metric.score}

    def contextual_relevancy(self) : 
        test_case=LLMTestCase(
        input=self.user_input, 
        actual_output=self.llm_output,
        retrieval_context=self.retrieval_context
        )
        metric = ContextualRelevancyMetric(threshold=0.5,model="gpt-4")
        metric.measure(test_case)
        return {"score": metric.score}




class FineTuningMetric(Evaluate) :  


    def hallucination(self,metric=HallucinationMetric) :  
        if not self.retrieval_context:
            return {
                "error": "Please upload a document to use the hallucination metric. This metric requires context to evaluate against."
            }
            
        test_case = LLMTestCase(
            input=self.user_input,
            actual_output=self.llm_output,
            context=self.retrieval_context
        )
        metric = HallucinationMetric(threshold=0.5,model="gpt-4")
        metric.measure(test_case)
        return {"score": metric.score}

    def toxicity(self) : 
        metric = ToxicityMetric(threshold=0.5,model="gpt-4")
        test_case = LLMTestCase(
        input=self.user_input,
        actual_output = self.llm_output
        )
        metric.measure(test_case)
        return {"score": metric.score}


class Summarization(Evaluate) : 

    def summarization(self) : 
        test_case = LLMTestCase(input=self.user_input, actual_output=self.llm_output)
        metric = SummarizationMetric(threshold=0.5,model="gpt-4")
        metric.measure(test_case)
        return {"score": metric.score}


