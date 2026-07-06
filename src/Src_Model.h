# ifndef Src_Model_h
# define Src_Model_h

# include <unistd.h>
# include <iostream>
# include <string>
# include <vector>
# include <map>

using namespace std;

class Src_Model {

    public :

        Src_Model();
        bool Init();
        ~Src_Model();

        // Morphology
        vector<string> MorType;
        map<string, int> MorMap;
        vector<int> MorNpar;
        vector<int> MorNDim;
        vector<string> MorFormula;
        vector<vector<string> > MorParname;
        // SED
        vector<string> SEDType;
        map<string, int> SEDMap;
        vector<int> SEDNpar;
        vector<string> SEDFormula;
        vector<vector<string> > SEDParname;

};

Src_Model::Src_Model(){}

Src_Model::~Src_Model(){}

bool Src_Model::Init(){

    // Morphology
    YAML::Node MorModel = YAML::LoadFile(Form("%s/src/Src_MorModel.yaml", cf.WorkDir.data()));
    int ntype = MorModel["Tag"].size();
    vector<string> parname;
    for (int ii=0;ii<ntype;ii++){

        MorType.push_back(MorModel["Tag"][ii].as<string>());
        if (MorModel[MorType[ii]]){
            MorMap.insert(pair<string, int>(MorType[ii], ii+1));
            MorNpar.push_back(MorModel[MorType[ii]]["Npar"].as<int>());
            MorNDim.push_back(MorModel[MorType[ii]]["NDim"].as<int>());
            MorFormula.push_back(MorModel[MorType[ii]]["Formula"].as<string>());

            parname.clear();
            for (int jj=0;jj<MorNpar[ii];jj++)
                parname.push_back(MorModel[MorType[ii]]["Parname"][jj].as<string>());

            MorParname.push_back(parname);

        }
        else{

            cerr<<"\033[31;1mError\033[0m : There is no definition of \""<<MorType[ii]<<"\" in "<<Form("%s/src/Src_MorModel.yaml", cf.WorkDir.data())<<"! Returned."<<endl;
            return -1;

        }
    }

    // SED
    YAML::Node SEDModel = YAML::LoadFile(Form("%s/src/Src_SEDModel.yaml", cf.WorkDir.data()));
    ntype = SEDModel["Tag"].size();
    for (int ii=0;ii<ntype;ii++){

        SEDType.push_back(SEDModel["Tag"][ii].as<string>());
        if (SEDModel[SEDType[ii]]){
            SEDMap.insert(pair<string, int>(SEDType[ii], ii+1));
            SEDNpar.push_back(SEDModel[SEDType[ii]]["Npar"].as<int>());
            SEDFormula.push_back(SEDModel[SEDType[ii]]["Formula"].as<string>());

            parname.clear();
            for (int jj=0;jj<SEDNpar[ii];jj++)
                parname.push_back(SEDModel[SEDType[ii]]["Parname"][jj].as<string>());

            SEDParname.push_back(parname);

        }
        else{

            cerr<<"\033[31;1mError\033[0m : There is no definition of \""<<SEDType[ii]<<"\" in "<<Form("%s/src/Src_SEDModel.yaml", cf.WorkDir.data())<<"! Returned."<<endl;
            return -1;

        }
    }
    parname.clear();
    parname.shrink_to_fit();

    return 0;

}

# endif
