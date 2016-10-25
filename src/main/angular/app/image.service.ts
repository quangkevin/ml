import { Injectable } from '@angular/core';
import { Http, Response } from '@angular/http';
import { Headers, RequestOptions } from '@angular/http';

import { Observable } from 'rxjs/Observable';
import { Image } from './image'
import { Matrix } from './matrix'
import { Tuple } from './tuple'

import * as _ from 'lodash';

@Injectable()
export class ImageService {
  constructor(private http: Http){}

  getImages(): Observable<Image[]> {
    return this.http.get('/service/image/count').map(this.extractImageData);
  }

  predictImage(image: Image): Observable<Tuple<Matrix>> {
    return this.http.get('/service/image/predict/' + image.id).map(this.extractImagePrediction);
  }

  private extractImageData(res: Response) {
    return _.range(res.json()).map(function(i) {
      return { url: 'service/image/' + i, id: i };
    });
  }

  private extractImagePrediction(res: Response) {
    return res.json();
  }
}
