/* 
 * File: src/bitmap.cpp
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Contains functionality for working with bitmap images.
 */

#include <stdio.h>
#include "bitmap.h"

using namespace std;


unsigned char *** reserveSpaceImage(int height, int width) {

    unsigned char *** image = new unsigned char **[height];
    for(int i = 0; i < height; i++){
        image[i] = new unsigned char * [width];
        for(int j = 0; j < width; j++){
            image[i][j] = new unsigned char[BYTES_PER_PIXEL];
        }
    }

    return image;
}



void freeSpaceImage(unsigned char ***image, int height, int width) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            delete[] image[i][j];
        }
        delete[] image[i];
    }
    delete[] image;
}



void generateBitmapImage (unsigned char*** image,
                          int height,
                          int width,
                          char* image_file_name) {

    int width_in_bytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = {0, 0, 0};
    int padding_size = (4 - (width_in_bytes) % 4) % 4;

    int stride = (width_in_bytes) + padding_size;

    FILE* image_file = fopen(image_file_name, "wb");

    unsigned char* file_header = createBitmapFileHeader(height, stride);
    fwrite(file_header, 1, FILE_HEADER_SIZE, image_file);

    unsigned char* info_header = createBitmapInfoHeader(height, width);
    fwrite(info_header, 1, INFO_HEADER_SIZE, image_file);

    for(int i = 0; i < height; i++){
        for(int j = 0; j < width; j++){
            fwrite(image[i][j],BYTES_PER_PIXEL,1,image_file);
            fwrite(padding, 1, padding_size, image_file);
        }
    }

    fclose(image_file);
}



unsigned char* createBitmapFileHeader (int height, int stride) {

    int file_size = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char file_header[] = {
        0,0,    
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
    };

    file_header[ 0] = (unsigned char)('B');
    file_header[ 1] = (unsigned char)('M');
    file_header[ 2] = (unsigned char)(file_size);
    file_header[ 3] = (unsigned char)(file_size >>  8);
    file_header[ 4] = (unsigned char)(file_size >> 16);
    file_header[ 5] = (unsigned char)(file_size >> 24);
    file_header[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return file_header;
}



unsigned char* createBitmapInfoHeader (int height, int width) {

    static unsigned char info_header[] = {
        0,0,0,0,          // header size
        0,0,0,0,          // image width
        0,0,0,0,          // image height
        0,0,              // number of color planes
        0,0,              // bits per pixel
        0,0,0,0,          // compression
        0,0,0,0,          // image size
        0,0,0,0,          // horizontal resolution
        0,0,0,0,          // vertical resolution
        0,0,0,0,          // colors in color table
        0,0,0,0,          // color count
    };

    info_header[0]  = (unsigned char)(INFO_HEADER_SIZE);
    info_header[4]  = (unsigned char)(width       );
    info_header[5]  = (unsigned char)(width  >>  8);
    info_header[6]  = (unsigned char)(width  >> 16);
    info_header[7]  = (unsigned char)(width  >> 24);
    info_header[8]  = (unsigned char)(height      );
    info_header[9]  = (unsigned char)(height >>  8);
    info_header[10] = (unsigned char)(height >> 16);
    info_header[11] = (unsigned char)(height >> 24);
    info_header[12] = (unsigned char)(1);
    info_header[14] = (unsigned char)(BYTES_PER_PIXEL*8);

    return info_header;
}

