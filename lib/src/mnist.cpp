#include "../include/mnist.h"

//This is some shit code i found online and fixed...

int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}


void ReadMNISTimage(std::string filename, int NumberOfImages, int DataOfAnImage,VECTOR2D &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
    std::ifstream file (filename,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<NumberOfImages;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    
                    arr.at(i).at((n_rows*r)+c) = ((double)temp / (double)255.0); //Normalised [0, 1]
                    //arr.at(i).at((n_rows*r)+c) = (((double)temp - (double)127.5) / (double)127.5); //Normalised [-1, 1]
                    //arr.at(i).at((n_rows*r)+c) = (double)temp;
                    
                }
            }
        }
    }
}

void ReadMNISTlabel(std::string filename, int NumberOfImages, int DataOfAnImage,VECTOR2D &arr)
{
    arr.resize(NumberOfImages,std::vector<double>(DataOfAnImage));
    std::ifstream file (filename,std::ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= ReverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= ReverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= ReverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= ReverseInt(n_cols);
        for(int i=0;i<NumberOfImages;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            //std::cout << arr[0].size() << " " << (n_rows*r)+c << "\n";
            //std::cout << (int)temp << "\n";
            arr.at(i).at(0) = (int)temp;
        }
    }
}