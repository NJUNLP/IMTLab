import os
import openai
from .imt_system import IMTSystem, logger

language_map = {"zh": "simplified Chinese", "en": "English", "de": "German"}

class ChatgptImt(IMTSystem):
    def __init__(self, args) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.MODEL = "gpt-3.5-turbo-0301"
        self.args = args
        self.src = language_map[args.src_lang]
        self.tgt = language_map[args.tgt_lang]

    def translate(self, src, template=None):
        if template is None:
            prompt = "Translate the following {} text to {}: {}".format(self.src, self.tgt, src)
        else:
            template_str = ""
            revised_hypo = template.revised_hypo[5:] if template.revised_hypo.startswith("<bos>") else template.revised_hypo
            tag = template.tag
            constraint = ""
            blank = False
            for i in range(len(tag)):
                if tag[i] == 0:
                    continue
                elif tag[i] < 4:
                    if blank:
                        template_str += " _"
                        blank = False
                    constraint += revised_hypo[i]
                else:
                    blank = True
                    if constraint:
                        template_str += constraint
                        constraint = ""
            if constraint:
                template_str += constraint
            elif blank:
                template_str += " _"
            else:
                logger.warning("template error!")
            
            #prompt 0
            prompt = "Translate the {0} sentence by filling in the {2} template. Strictly follow the given {2} template and generate a whole translation\n{0} sentence: {1}\n{2} template: {3}\n{2} translation:".format(self.src, src, self.tgt, template_str.strip())

            #prompt 1
            # prompt = "Strictly follow the provided {2} template and information to generate a grammatically correct {2} sentence that accurately conveys the same meaning as the given {0} sentence. You must generate a complete sentence and any deviation from the template should be avoided.\n{0} sentence: {1}\n{2} template: {3}\n{2} sentence:".format(self.src, src, self.tgt, template_str.strip())

            #prompt 2
            # prompt = "Use the provided {2} template and information to generate a sentence in {2} that conveys the same meaning as the given {0} sentence. Ensure that the sentence follows the given template exactly.\n{0} sentence: {1}\n{2} template: {3}\nComplete {2} sentence:".format(self.src, src, self.tgt, template_str.strip())

            #prompt 3
            # prompt = "{0} sentence: {1}\n{2} template: {3}\nCreate a {2} sentence using the given template and information that accurately translates the provided {0} sentence. You must conform to the template and generate the whole translation.\n{2} sentence:".format(self.src, src, self.tgt, template_str.strip())

            #prompt 4
            # prompt = "{0} sentence: {1}\n{2} template: {3}\nYour task is to provide a German translation of the given English sentence. You must use the given {2} template and information exactly as provided without making any changes, and generate a complete translation.\n{2} translation:".format(self.src, src, self.tgt, template_str.strip())

        response = openai.ChatCompletion.create(
            model=self.MODEL,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0,
            max_tokens=200,
        )
        hypo = response["choices"][0]["message"]["content"]
        total_tokens = response["usage"]["total_tokens"]
        logger.debug("prompt:\n{}".format(prompt))
        logger.debug("hypothesis:\n{}".format(hypo))
        logger.debug("token used: {}".format(total_tokens))
        return hypo
