import json
import argparse

class ParseGCV:
    def __init__(self, data, debug=False):
        self.data = data
        self.debug = debug

    # face bouding box
    def get_fdBoundingPoly(self):
        self._get_vertices(target='fdBoundingPoly')

    def get_BoundingPoly(self):
        self._get_vertices(target='BoundingPoly')

    def _get_vertices(self, target):
        data = self.data

        out = []
        try:
            vertices = data['faceAnnotations'][0][target]['vertices']
            #out = [ (v['x'], v['y']) for v in vertices ]
            out = [ (vertices[0]['x'], vertices[0]['y']),
                    (vertices[2]['x'], vertices[2]['y']) ]
            if (self.debug):
                entry = {'veritices-'+target: out}
                print(json.dumps(entry))
        except:
            if (self.debug):
                print('bouding box detection failure ', target)


        return out

    def get_landmarks(self):
        data = self.data

        out = dict()
        try:
            landmarks = data['faceAnnotations'][0]['landmarks']

            for l in landmarks:
                out[l['type']] = (l['position']['x'], l['position']['y'])
            if (self.debug):
                print(json.dumps(out))
        except:
            if (self.debug):
                print('face keypoint detection failure')

        return out

    def get_labels(self, threshold=None):
        data = self.data

        labels = data['labelAnnotations']
        out = dict()
        for l in labels:
            if (threshold and l['score'] < threshold):
                next
            out[l['description']] = (l['score'])

        if (self.debug):
            print(json.dumps(out))
        return out
    #print(data.faceAnnotations[0].fdBoundingPoly)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse face by google cloud vision api')
    #args = parser.add_argument('img', type=argparse.FileType('r'), help='image file path')
    args = parser.add_argument('files', nargs='+', help='image files')
    args = parser.add_argument('--debug', action='store_true', help='debug flag')
    #args = parser.add_argument('--mkdir', action='store_true', help='mkdir outfile if not exist')
    #args = parser.add_argument('--overwrite', action='store_true', help='overwrite outfilefile if exist')
    #args = parser.add_argument('--category', default='test')
    args = parser.parse_args()

    for filename in args.files:
        f = open(filename)
        for line in f:
            data = json.loads(line)
            parser = ParseGCV(data, args.debug)
            parser.get_fdBoundingPoly()
            parser.get_landmarks()
            parser.get_labels()
            #print(json.dumps(data))
