
import os,json


def convert(raw_file, line_num=None):
    assert os.path.exists(raw_file), "raw file is not exists!"
    with open(raw_file, "r", encoding="utf-8") as file:
        dis_file = open("./instraction_train_data.json", "w", encoding="utf-8")
        data = eval(file.read())
        c = 0
        for item in data:
            if line_num and c >= line_num:
                break
            if "input" in item and "instruction" in item and "output" in item:
                dic = dict()
                dic["history"] = [item["input"]]
                dic["prompt"] = item["instruction"]
                dic["response"] = item["output"]
                dis_file.write(json.dumps(dic, ensure_ascii=False)+"\n")
                c += 1


if __name__ == '__main__':
    filename = "./raw.json"
    convert(filename)

