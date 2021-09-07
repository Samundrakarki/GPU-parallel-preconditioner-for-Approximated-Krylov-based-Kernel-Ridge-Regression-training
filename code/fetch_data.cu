#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "fetch_data.h"

void fetch_data_energy_efficiency(double* input_data, double* output_data,  int data_size, int input_dim, int output_dim)
{
    FILE* stream = fopen("ENB2012_data.csv", "r");
    if(stream == NULL)
    {
        fprintf(stderr, "Error in loading Energy efficiency data file");
        return;
    }

    char line[1024];

    int skip = 0;
    int n_data = 0;
    
    while (fgets(line, 1024, stream))
    {
        char* temp = strdup(line);
        if (skip > 0)
        {
            char* tok = strtok(temp, ",");
            int dim = 0;
            int out_dim = 0;
            int in_dim = 0;
            while(tok != NULL)
            {
                if(dim < input_dim)
                {
                    input_data[n_data + (in_dim*data_size)] = atof(tok);
                    in_dim++;
                }else{
                    output_data[n_data + (data_size*out_dim)] = atof(tok);
                    out_dim++;
                }
                tok = strtok(NULL, ",");
                dim++;
            }
            n_data++;
        }
        skip = 1;
    }
}

void read_numpy_files(FILE* fp, double* vector, int data_size, int dim)
{

    const int size_magic = 6;
    const int size_version = 2;
    const int size_header_len = 2;

    int byte_read;
    long size_of_file;
    int err;
    int start_of_data;
    uint16_t header_len_value;

    unsigned char* magic_number;
    magic_number = (unsigned char*) malloc( size_magic * sizeof(unsigned char));
    unsigned char standard_magic_number[] = {0x93,'N','U','M','P','Y'}; 
    
    byte_read = fread(magic_number, sizeof(unsigned char), 6, fp);
    
    if(!byte_read)
    {
        fprintf(stderr, "Error: File read operation.\n");
        return;
    }

    for(int i =0; i< size_magic; i++)
    {
        if(magic_number[i] != standard_magic_number[i])
        {
            fprintf(stderr, "Error: Not a numpy file");
            return;
        }
    }
    
    err = fseek(fp, size_magic + size_version, SEEK_SET);
    if(err == -1)
    {
        fprintf(stderr, "Error on fseek: Problem with moving the pointer");
        return;
    }
    
    byte_read = fread(&header_len_value, sizeof(uint16_t), 1, fp);
    if(!byte_read)
    {
        fprintf(stderr, "Error: File read operation.\n");
        return;
    }

    char* dict = (char*) malloc((header_len_value + 1) * sizeof(char));
    err = fseek(fp, size_magic + size_version + size_header_len, SEEK_SET);
    if(err == -1)
    {
        fprintf(stderr, "Error on fseek: Problem with moving the pointer");
        return;
    }
    
    byte_read = fread(dict, sizeof(char), header_len_value, fp);
    if(!byte_read)
    {
        fprintf(stderr, "Error: File read operation.\n");
        return;
    }
    dict[header_len_value] = '\0';

    printf("%s ", dict);

    fseek(fp, 0, SEEK_END);
    size_of_file = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    start_of_data = size_magic + size_version + size_header_len + header_len_value;
    err = fseek(fp, start_of_data, SEEK_SET);

    byte_read = fread(vector, sizeof(double), (size_of_file - start_of_data)/sizeof(double), fp);
    if(!byte_read)
    {
        fprintf(stderr, "Error: File read operation.\n");
        return;
    }
}

void fetch_data_QM9(double* input_data, double* output_data, int data_size, int input_dim, int output_dim)
{
    FILE* stream_input = fopen("/home/pzaspel/data/qm9_X.npy", "rb");
    FILE* stream_output = fopen("/home/pzaspel/data/qm9_Y.npy", "rb");

    if(stream_input == NULL || stream_output == NULL)
    {
        fprintf(stderr, "Error in loading the files");
        return;
    }

    read_numpy_files(stream_input, input_data, data_size, input_dim);
    read_numpy_files(stream_output, output_data, data_size, output_dim);

    fclose(stream_input);
    fclose(stream_output);

}
 
void fetch_data_MD17(double* input_data, double* output_data, int data_size, int input_dim, int output_dim)
{
    FILE* stream_input = fopen("/home/pzaspel/data/md17_X.npy", "rb");
    FILE* stream_output = fopen("/home/pzaspel/data/md17_Y.npy", "rb");

    if(stream_input == NULL || stream_output == NULL)
    {
        fprintf(stderr, "Error in loading the files");
        return;
    }

    read_numpy_files(stream_input, input_data, data_size, input_dim);
    read_numpy_files(stream_output, output_data, data_size, output_dim);

    fclose(stream_input);
    fclose(stream_output);

}

// int main()
// {
//     double* input_data;
//     double* output_data;

//     int data_size = 133885;
//     int input_dim = 11960;
//     int output_dim = 1;
    

//     int data_size_1 = 993237;
//     int input_dim_1 = 3121;
//     int output_dim_1 = 1;

//     input_data = (double* ) malloc(data_size * input_dim*sizeof(double));
//     output_data = (double* ) malloc(data_size * output_dim*sizeof(double));


//     fetch_data_QM9(input_data, output_data, data_size, input_dim, output_dim);

//     input_data = (double* ) malloc(data_size_1 * input_dim_1*sizeof(double));
//     output_data = (double* ) malloc(data_size_1 * output_dim_1*sizeof(double));

//     fetch_data_MD17(input_data, output_data, data_size, input_dim, output_dim);

//     free(input_data);
//     free(output_data);
// }