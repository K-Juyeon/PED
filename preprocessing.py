import os
import shutil
import splitfolders

dataset = r"F:\PED\labeleddata"
assembel = r"F:\PED\asembledata"
splitdata = r"F:\PED\splitdata"

class preprocessing() :
    def assemble(data) :
        for paths, dirs, files in os.walk(dataset) :
            pathslist = os.listdir(paths)
            filelist = [file for file in pathslist if file.endswith(".jpg") or file.endswith(".png")]

            for file in filelist :
                name_list = paths.split('\\')
                del name_list[0:3]
                new_path = '_'.join(name_list)
                if not os.path.exists(os.path.join(assembel, new_path)) :
                    os.makedirs(os.path.join(assembel, new_path))
                shutil.copy2(os.path.join(paths, file), os.path.join(assembel, new_path, file))
                print(os.path.join(assembel, new_path, file))

    def split(data, output_data) :
        if not os.path.exists(splitdata) :
            os.makedirs(splitdata)
        splitfolders.ratio(data, output=output_data, seed=1337, ratio=(.6, .2, .2)) #, group_prefix=2(.jpg, .json)

        for paths, dirs, files in os.walk(splitdata) :
            pathslist = os.listdir(paths)
            filelist = [file for file in pathslist if file.endswith(".jpg")]
            if len(filelist) > 1 :
                print(paths.split('\\')[-2], paths.split('\\')[-1], ":", len(filelist))

    def puttogether(data) :
        path = os.listdir(data)
        for li in path :
            if "무" in li :
                pass
            else :
                if not os.path.exists(os.path.join(splitdata, li, "개_안구초음파_무")) :
                    os.makedirs(os.path.join(splitdata, li, "개_안구초음파_무"))
                if not os.path.exists(os.path.join(splitdata, li, "개_일반_무")) :
                    os.makedirs(os.path.join(splitdata, li, "개_일반_무"))
                if not os.path.exists(os.path.join(splitdata, li, "고양이_일반_무")) :
                    os.makedirs(os.path.join(splitdata, li, "고양이_일반_무"))

        for li in path :
            if "무" in li :
                for paths, dirs, files in os.walk(splitdata, li) :
                    if "개" in paths and "안구초음파" in paths :
                        pass


                
                    
                    
            


if __name__ == "__main__":
    # preprocessing.assemble(dataset)
    # preprocessing.split(assembel, splitdata)
    preprocessing.puttogether(splitdata)