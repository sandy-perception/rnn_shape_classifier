import glob
import os



def find_files(path):
    """
    Find files at given path.

    :param path: path to find files.
    :return: list of files
    """
    regex = os.path.join(path,"*.txt")
    return glob.glob(regex)

def get_lines(file_path):
    """
    Read input files
    :param file_path: get lines from data
    :return: lines from input files
    """
    lines = open(file_path, encoding='utf-8').read().strip().split('\n')
    return [line for line in lines]


def get_training_data(annotation_dir):
    """
    Build the category_lines dictionary, a list of names per language

    :param annotation_dir:
    :return: all_lines, category (class) list, class data dictionary
    """

    all_lines = []
    all_categories = []
    category_lines = {}

    for filename in find_files(annotation_dir):
        f_name = os.path.splitext(os.path.basename(filename))[0]
        category = f_name.split("_")[0]
        all_categories.append(category)
        lines = get_lines(filename)
        category_lines[category] = lines
        all_lines.extend(lines)

    return all_lines, all_categories, category_lines


def get_instruction_set(size=128):
    """
    Function to define rnn word set. Integer range equally distributed around 0.
    Total size would be integers plus comma

    :param size: size of instruction set
    :return: instruction set
    """

    limit = int((size-1) / 2)
    instr_list = list()
    instr_list.extend([','])
    instr_list.extend([i for i in range(-limit,limit+1)])
    
    return instr_list


def update_instruction_set(instr_list,lines) :
    """
    update instruction set by adding characters which are new in input lines.

    :param instr_list: original instruction list
    :param lines: lines of whole data set
    :return: updated instruction list
    """

    line_no = 0
    outlier_set = set()
    outlier_occurances = 0
    for line in lines :
        line_no+=1
        vects=line.split(";")
       
        for vec in vects[1:-1]: # skiping first and last coordinate set
            
            instrs = vec_to_instructions(vec)
            for instr in instrs :                   
                if not instr in instr_list :
                    outlier_set.add(instr)
                    outlier_occurances+=1
                    instr_list.append(instr)
              
    print("outliers set size",len(outlier_set)," # of occurances" , outlier_occurances)
    print("outliers ", outlier_set)
    
    return instr_list


def vec_to_instructions(vec):
    """
    Convert vector string to set of instruction : (-2,3) to [-2,',',3] #mix of int and chars
    :param vec: 2d integer vector
    :return:
    """
    instrs = vec[1:-1].split(",") # remove brackets and split

    instructions = list()
    instructions.append(int(instrs[0]))
    instructions.append(int(instrs[1]))
    instructions.append(',')

    return instructions