import { Component, OnInit } from '@angular/core';

import './rxjs-operators';

import { ImageService } from  './image.service';
import { Image } from './image';
import { Matrix } from './matrix'
import { Tuple } from './tuple'

@Component({
  moduleId: module.id,
  selector: 'my-app',
  templateUrl: 'app.component.html',
  providers: [ ImageService ]
})

export class AppComponent implements OnInit {
  images: Image[];
  visibleImages: Image[];
  pageIndex: number = 0;
  pageSize: number = 225;
  prediction: Tuple<Matrix>;

  constructor(private imageService: ImageService) {}

  ngOnInit() {
    this.getImages();
  }

  getImages() {
    this.imageService.getImages().subscribe(images => {
      this.images = images;
      this.sliceImages();
    });
  }

  previousPage() {
    this.pageIndex = Math.max(0, this.pageIndex - 1);
    this.sliceImages();
  }

  nextPage() {
    this.pageIndex = Math.min(this.images.length/this.pageSize, this.pageIndex + 1);
    this.sliceImages();
  }

  predict(image: Image) {
    this.imageService.predictImage(image).subscribe(result => {
      this.prediction = result;
    });
  }

  getResult(prediction: Tuple<Matrix>): number {
    var max = 0;

    for (var i = 0; i < prediction.y.val.length; ++i) {
      if (prediction.y.val[max][0] < prediction.y.val[i][0]) {
        max = i;
      }
    }

    return max;
  }

  private sliceImages() {
    this.visibleImages = this.images.slice(this.pageIndex * this.pageSize,
                                           Math.min(this.pageIndex * this.pageSize + this.pageSize,
                                                    this.images.length));
  }
}
