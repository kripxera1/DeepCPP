/* 
 * File: include/bitmap.h
 * Author: Antonio Manuel Escudero Vargas <antoniomanuelescuderovargas@gmail.com>
 * License: MIT
 * Description: Provides functionality for working with bitmap images.
 */

#ifndef BITMAP
#define BITMAP

const int BYTES_PER_PIXEL = 3;
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

void freeSpaceImage(unsigned char ***image, int height, int width);

unsigned char *** reserveSpaceImage(int height,int width);

void generateBitmapImage(unsigned char*** image, int height, int width, char* imageFileName);

unsigned char* createBitmapFileHeader(int height, int stride);

unsigned char* createBitmapInfoHeader(int height, int width);

#endif