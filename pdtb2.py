
import sys
import csv
import re
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

class CorpusReader:
    def __init__(self, src_filename):
        self.src_filename = src_filename
   
    def iter_data(self, display_progress=True):
        with open(self.src_filename) as f:
            row_iterator = csv.reader(f)
            next(row_iterator)
            for i, row in enumerate(row_iterator):
                if display_progress:
                    sys.stderr.write("\r")
                    sys.stderr.write("row {} ".format(i+1))
                    sys.stderr.flush()
                yield Datum(row)
            if display_progress: sys.stderr.write("\n")

class Datum:
    header = [
        # 语料库
        'Relation', 'Section', 'FileNumber',
        ##### 连接词.
        'Connective_SpanList', 'Connective_GornList', 'Connective_Trees',
        'Connective_RawText', 'Connective_StringPosition',
        'SentenceNumber', 'ConnHead', 'Conn1', 'Conn2', 'ConnHeadSemClass1',
        'ConnHeadSemClass2', 'Conn2SemClass1', 'Conn2SemClass2',
        # 连接词属性.
        'Attribution_Source', 'Attribution_Type', 'Attribution_Polarity',
        'Attribution_Determinacy',
        'Attribution_SpanList', 'Attribution_GornList', 'Attribution_Trees',
        'Attribution_RawText',
        ##### Arg1
        'Arg1_SpanList', 'Arg1_GornList', 'Arg1_Trees', 'Arg1_RawText',
        # Arg1 属性.
        'Arg1_Attribution_Source', 'Arg1_Attribution_Type', 'Arg1_Attribution_Polarity',
        'Arg1_Attribution_Determinacy',
        'Arg1_Attribution_SpanList', 'Arg1_Attribution_GornList', 'Arg1_Attribution_Trees',
        'Arg1_Attribution_RawText',
        ##### Arg2.
        'Arg2_SpanList', 'Arg2_GornList', 'Arg2_Trees', 'Arg2_RawText',
        # Arg2 属性.
        'Arg2_Attribution_Source', 'Arg2_Attribution_Type', 'Arg2_Attribution_Polarity',
        'Arg2_Attribution_Determinacy',
        'Arg2_Attribution_SpanList', 'Arg2_Attribution_GornList', 'Arg2_Attribution_Trees',
        'Arg2_Attribution_RawText',
        ##### Sup1.
        'Sup1_SpanList', 'Sup1_GornList', 'Sup1_Trees', 'Sup1_RawText',
        ##### Sup2
        'Sup2_SpanList', 'Sup2_GornList', 'Sup2_Trees', 'Sup2_RawText',
        ##### 完整的 raw text.
        'FullRawText']
    
    def __init__(self, row):
        """        
        若 row 是列表则直接处理；
        若 row 是字符串则使用 csv.reader 将其解析为列表；
        属性名称由类变量 Datum.header 中给出。
        """
        if row.__class__.__name__ in ('str', 'unicode'):
            row = list(csv.reader([row.strip()]))[0]        
        for i in range(len(row)):
            att_name = Datum.header[i]
            row_value = row[i]
            if re.search(r"SpanList", att_name):
                row_value = self.__process_span_list(row_value)
            elif re.search(r"GornList", att_name):
                row_value = self.__process_gorn_list(row_value)
            elif att_name in ('Connective_StringPosition', 'SentenceNumber'):
                if row_value:
                    row_value = int(row_value)
                else:
                    row_value = None
            elif re.search(r"Trees", att_name):
                row_value = self.__process_trees(row_value)            
            elif not row_value:
                row_value = None
            setattr(self, att_name, row_value)

    # SEMANTIC VALUES 语义值
    def primary_semclass1(self):
        """ Comparison, Contingency, Expansion, Temporal """
        return self.semclass1_values()[0]

    def secondary_semclass1(self):
        """
        Alternative, Asynchronous, Cause, Concession, Condition,
        Conjunction, Contrast, Exception, Instantiation, List,
        Pragmatic cause, Pragmatic concession, Pragmatic condition,
        Pragmatic contrast, Restatement, Synchrony
        """
        vals = self.semclass1_values()
        if len(vals) >= 2:
            return vals[1]
        else:            
            return None

    def tertiary_semclass1(self):
        """
        Chosen alternative, Conjunctive, Contra-expectation,
        Disjunctive, Equivalence, Expectation, Factual past, Factual
        present, General, Generalization, Hypothetical, Implicit
        assertion, Justification, Juxtaposition, NONE, Opposition,
        Precedence, Reason, Relevance, Result,Specification,
        Succession, Unreal past, Unreal present
        """
        vals = self.semclass1_values()
        if len(vals) >= 3:
            return vals[2]
        else:            
            return None    

    def semclass1_values(self):
        if self.ConnHeadSemClass1:
            return self.ConnHeadSemClass1.split(".")
        else:
            return [None]

    ######################################################################
    # TOKENIZING AND POS-TAGGING WITH THE OPTION TO CONVERT
    # CONVERTABLE TAGS TO WORDNET STYLE

    def arg1_words(self, lemmatize=False):
        """ 返回与 Arg1 关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.arg1_pos, lemmatize=lemmatize)

    def arg2_words(self, lemmatize=False):
        """ 返回与 Arg2 关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.arg2_pos, lemmatize=lemmatize)

    def arg1_attribution_words(self, lemmatize=False):
        """ 返回与 Arg1 的属性关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.arg1_attribution_pos, lemmatize=lemmatize)

    def arg2_attribution_words(self, lemmatize=False):
        """ 返回与 Arg2 的属性关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.arg2_attribution_pos, lemmatize=lemmatize)

    def connective_words(self, lemmatize=False):
        """ 返回与 Explicit 或 AltLex 连接词关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.connective_pos, lemmatize=lemmatize)

    def sup1_words(self, lemmatize=False):
        """ 返回与 Sup1 关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.sup1_pos, lemmatize=lemmatize)

    def sup2_words(self, lemmatize=False):
        """ 返回与 Sup2 关联的单词列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.__words(self.sup2_pos, lemmatize=lemmatize)

    def __words(self, method, lemmatize=False):
        """ X_words 函数用来获取单词的内部方法。"""
        lemmas = method(lemmatize=lemmatize)
        return [x[0] for x in lemmas]
        
    def arg1_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Arg1 关联的 (word, pos) 对列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.arg_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def arg2_pos(self,  wn_format=False, lemmatize=False):
        """ 返回与 Arg2 关联的 (word, pos) 对列表. lemmatize=True. 在列表中使用 nltk.stem.WordNetStemmer(). """
        return self.arg_pos(2,  wn_format=wn_format, lemmatize=lemmatize)

    def arg_pos(self, index, wn_format=False, lemmatize=False):
        """ 返回与 ArgN 关联的 (word, pos) 对列表，其中 N = 索引（1 或 2）"""
        return self.__pos("Arg%s" % index, wn_format=wn_format, lemmatize=lemmatize)

    def arg1_attribution_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Arg1 关联的 (word, pos) 对列表 """
        return self.arg_attribution_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def arg2_attribution_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Arg2 关联的 (word, pos) 对列表 """
        return self.arg_attribution_pos(2, wn_format=wn_format, lemmatize=lemmatize)

    def arg_attribution_pos(self, index, wn_format=False, lemmatize=False):
        """ 返回与 ArgN 的属性关联的 (word, pos) 对列表 """
        return self.__pos("Arg%s_Attribution" % index, wn_format=wn_format, lemmatize=lemmatize)

    def connective_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Explicit 或 AltLex 连接词关联的 (word, pos) 对列表 """
        return self.__pos("Connective", wn_format=wn_format, lemmatize=lemmatize)

    def sup1_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Sup1 关联的 (word, pos) 对列表 """
        return self.sup_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def sup2_pos(self, wn_format=False, lemmatize=False):
        """ 返回与 Sup2 关联的 (word, pos) 对列表 """
        return self.sup_pos(2, wn_format=wn_format, lemmatize=lemmatize)

    def sup_pos(self, index, wn_format=False, lemmatize=False):
        """ 返回与 SupN 关联的 (word, pos) 列表，其中 N = 索引（1 或 2）。"""
        return self.__pos("Sup%s" % index, wn_format=wn_format, lemmatize=lemmatize)

    def __pos(self, typ, wn_format=False, lemmatize=False):
        """ 用于获取 POS 的内部方法 """
        results = []
        trees = eval("self.%s_Trees" % typ)
        for tree in trees:
            results += tree.pos()
        if lemmatize:
            results = list(map(self.__treebank2wn_pos, results))
            results = list(map(self.__lemmatize, results))
        elif wn_format:
            results = list(map(self.__treebank2wn_pos, results))
        return results

    def __treebank2wn_pos(self, lemma):
        """ 将词条的标签转换为 WordNet 格式的内部方法 """
        string, tag = lemma
        tag = tag.lower()
        if tag.startswith('v'):
            tag = 'v'
        elif tag.startswith('n'):
            tag = 'n'
        elif tag.startswith('j'):
            tag = 'a'
        elif tag.startswith('rb'):
            tag = 'r'
        return (string, tag)

    def __lemmatize(self, lemma):
        """ 将 nltk.stem.WordNetStemmer() 应用于 (word, pos) 的内部方法 """
        string, tag = lemma
        if tag in ('a', 'n', 'r', 'v'):        
            wnl = WordNetLemmatizer()
            string = wnl.lemmatize(string, tag)
        return (string, tag)

    ######################################################################    
    # POSITIONING 定位.

    def relative_arg_order(self):
        """
        1S ... 1F ... 2s ... 2f -> arg1_precedes_arg2
        1S ... 2s ... 2f ... 1F -> arg1_contains_arg2
        1S ... 2s ... 1F ... 2f -> arg1_precedes_and_overlaps_but_does_not_contain_arg2
        2S ... 2F ... 1S ... 1F -> arg2_precedes_arg1
        2S ... 1S ... 1F ... 2F -> arg2_contains_arg1
        2S ... 1S ... 2F ... 2F -> arg2_precedes_and_overlaps_but_does_not_contain_arg1
        """
        arg1_indices =  [i for span in self.Arg1_SpanList for i in span]
        arg1_start = min(arg1_indices)
        arg1_finish = max(arg1_indices)
        arg2_indices = [i for span in self.Arg2_SpanList for i in span]      
        arg2_start = min(arg2_indices)
        arg2_finish = max(arg2_indices)
        # Arg1 在前面:
        if arg1_finish < arg2_start:
            return 'arg1_precedes_arg2'
        if arg1_start < arg2_start and arg2_finish < arg1_finish:
            return 'arg1_contains_arg2'
        if arg1_start < arg2_start and arg2_start < arg1_finish and arg1_finish < arg2_finish:
            return 'arg1_precedes_and_overlaps_but_does_not_contain_arg2'
        # Arg2 在前面:
        if arg2_finish < arg1_start:
            return 'arg2_precedes_arg1'
        if arg2_start < arg1_start and arg1_finish < arg2_finish:
            return 'arg2_contains_arg1'
        if arg2_start < arg2_start and arg1_start < arg2_finish and arg2_finish < arg1_finish:
            return 'arg2_precedes_and_overlaps_but_does_not_contain_arg1'
        raise Exception("No relation could be determined for the two arguments!\n%s" % self.FullRawText)

    def arg1_precedes_arg2(self):
        """ 如果整个 Arg1 在整个 Arg2 之前则返回 True: 1S ... 1F ... 2s ... 2f """
        if self.relative_arg_order() == 'arg1_precedes_arg2':
            return True
        else:
            return False

    def arg2_precedes_arg1(self):
        """ 如果整个 Arg2 在整个 Arg1 之前则返回 True: 2S ... 2F ... 1S ... 1F """
        if self.relative_arg_order() == 'arg2_precedes_arg1':
            return True
        else:
            return False

    def arg1_contains_arg2(self):
        """ 如果 Arg1 完全包含 Arg2 则返回 True: 1S ... 2s ... 2f ... 1F """
        if self.relative_arg_order() == 'arg1_contains_arg2':
            return True
        else:
            return False

    def arg2_contains_arg1(self):
        """ 如果 Arg2 完全包含 Arg1 则返回 True: 2S ... 1S ... 1F ... 2F """
        if self.relative_arg_order() == 'arg2_contains_arg1':
            return True
        else:
            return False

    def arg1_precedes_and_overlaps_but_does_not_contain_arg2(self):
        """ Arg1 在 Arg2 之前开始, 也在 Arg2 之前结束: 1S ... 2s ... 1F ... 2f """
        if self.relative_arg_order() == 'arg1_precedes_and_overlaps_but_does_not_contain_arg2':
            return True
        else:
            return False 

    def arg2_precedes_and_overlaps_but_does_not_contain_arg1(self):
        """  Arg2 在 Arg1 之前开始, 也在 Arg1 之前结束: 2S ... 1S ... 2F ... 2F """
        if self.relative_arg_order() == 'arg2_precedes_and_overlaps_but_does_not_contain_arg1':
            return True
        else:
            return False        

    ######################################################################
    # NORMALIZATION 正则化.

    def conn_str(self, distinguish_implicit=True):
        """ 提供一种方法来查看连接词的直观主元素 """
        rel = self.Relation
        if rel == 'Explicit':
            return self.ConnHead
        elif rel == 'AltLex':
            return self.Connective_RawText
        elif rel == 'Implicit':
            prefix = ""
            if distinguish_implicit:
                prefix = "Implicit="            
            return prefix + self.Conn1
        else:
            return None

    def final_arg1_attribution_source(self):
        """ 遵循 Arg1 的继承属性值 """
        return self.final_arg_attribution_source(1)

    def final_arg2_attribution_source(self):
        """ 遵循 Arg2 的继承属性值 """
        return self.final_arg_attribution_source(2)

    def final_arg_attribution_source(self, index):
        """ 若参数的属性是 Inh（继承）则提供来自连接词的继承值 """
        if index not in (1,2):
            raise ArgumentError('index must be int 1 or int 2; was %s (type %s).\n' % (index, index.__class__.__.name__))
        src = eval("self.Arg%s_Attribution_Source" % index)
        if src == "Inh":
            src = self.Attribution_Source
        return src

    ######################################################################
    # INTERNAL HELPER METHODS 内部辅助方法.

    def __process_span_list(self, s):
        if not s:
            return []
        parts = re.split(r"\s*;\s*", s)
        seqs = list(map((lambda x : map(int, re.split(r"\s*\.\.\s*", x))), parts))
        return seqs

    def __process_gorn_list(self, s):
        if not s:
            return []
        parts = re.split(r"\s*;\s*", s)
        seqs = map((lambda x : map(int, re.split(r"\s*,\s*", x))), parts)
        return seqs

    def __process_trees(self, s):
        if not s:
            return []
        tree_strs = s.split("|||")
        return [Tree.fromstring(s) for s in tree_strs]
            

