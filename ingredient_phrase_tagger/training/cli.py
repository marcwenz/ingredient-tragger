import re
import decimal
import optparse
import pandas as pd

from . import utils


class Cli(object):
    def __init__(self, count=-1, offset=0, data_path='nyt-ingredients-dropnan.csv'):
        """
        This generates the training and test data.
        argv:
        count (int) - number of samples, if -1 takes whole dataset
        offset (int) - count from the top
        data-path (str) - path to data
        """
        self.count = count
        self.offset = offset
        self.data_path = data_path
        self._upstream_cursor = None

    def run(self):
        return self.generate_data(self.count, self.offset)

    def generate_data(self, count, offset):
        """
        Generates training data in the CRF++ format for the ingredient
        tagging task
        """
        df = pd.read_csv(self.data_path, index_col="index")
        df = df.fillna("")
        # Drop rows with no data
        # df[df['input'].map(len) != 0].reset_index(drop=True).to_csv("nyt-ingredients-dropnan.csv")

        start = int(offset)
        if count == -1:
            end = -1
        else:
            end = int(offset) + int(count)

        df_slice = df.iloc[start: end]

        cL = []

        for index, row in df_slice.iterrows():
            try:
                cI = []
                # extract the display name
                display_input = utils.cleanUnicodeFractions(row["input"])
                tokens = utils.tokenize(display_input)
                # print([tok for tok in tokens])
                del(row["input"])

                # print(display_input)
                # print([tok for tok in tokens.__iter__()])
                rowData = self.addPrefixes(
                    [(t, self.matchUp(t, row)) for t in tokens])
                # print(rowData)

                fL = []
                for i, (token, tags) in enumerate(rowData):
                    features = utils.getFeatures(token, i+1, tokens)
                    fL.append([token, features, self.bestTag(tags)])
                    # print(features)
                    # print(utils.joinLine([token] + features + [self.bestTag(tags)]))
                    # From here on we have the full feature extraction process complete.
                cI.extend((fL, display_input))
                cL.append(cI)
                # print()

            # ToDo: deal with this
            except UnicodeDecodeError:
                pass

            # print()

        return cL

    def parseNumbers(self, s):
        """
        Parses a string that represents a number into a decimal data type so that
        we can match the quantity field in the db with the quantity that appears
        in the display name. Rounds the result to 2 places.
        This is needed for processing the input data to match the tag and number, as tags are stored in int format (see matchUp)
        """
        ss = utils.unclump(s)
        # TODO: For numbers as word (one), substitute the digit
        m3 = re.match('^\d+$', ss)
        if m3 is not None:
            return decimal.Decimal(round(float(ss), 2))

        m1 = re.match(r'(\d+)\s+(\d)/(\d)', ss)
        if m1 is not None:
            num = int(m1.group(1)) + (float(m1.group(2)) / float(m1.group(3)))
            return decimal.Decimal(str(round(num, 2)))

        m2 = re.match(r'^(\d)/(\d)$', ss)
        if m2 is not None:
            num = float(m2.group(1)) / float(m2.group(2))
            return decimal.Decimal(str(round(num, 2)))

        return None

    def matchUp(self, token, ingredientRow):
        """
        Returns our best guess of the match between the tags and the
        words from the display text.

        This problem is difficult for the following reasons:
            * not all the words in the display name have associated tags
            * the quantity field is stored as a number, but it appears
              as a string in the display name
            * the comment is often a compilation of different comments in
              the display name

        """
        ret = []

        # strip parens from the token, since they often appear in the
        # display_name, but are removed from the comment.
        token = utils.normalizeToken(token)
        decimalToken = self.parseNumbers(token)

        # from past.builtins import basestring
        for key, val in ingredientRow.iteritems():
            if isinstance(val, str):  # str?

                for n, vt in enumerate(utils.tokenize(val)):
                    if utils.normalizeToken(vt) == token:
                        ret.append(key.upper())

            elif decimalToken is not None:
                try:
                    if val == decimalToken:
                        ret.append(key.upper())
                except:
                    pass

        return ret

    def addPrefixes(self, data):
        """
        We use BIO tagging/chunking to differentiate between tags
        at the start of a tag sequence and those in the middle. This
        is a common technique in entity recognition.

        Reference: http://www.kdd.cis.ksu.edu/Courses/Spring-2013/CIS798/Handouts/04-ramshaw95text.pdf
        """
        prevTags = None
        newData = []

        for n, (token, tags) in enumerate(data):

            newTags = []

            for t in tags:
                p = "B" if ((prevTags is None) or (t not in prevTags)) else "I"
                newTags.append("%s-%s" % (p, t))

            newData.append((token, newTags))
            prevTags = tags

        return newData

    def bestTag(self, tags):

        if len(tags) == 1:
            return tags[0]

        # if there are multiple tags, pick the first which isn't COMMENT
        else:
            for t in tags:
                if (t != "B-COMMENT") and (t != "I-COMMENT"):
                    return t

        # we have no idea what to guess
        return "OTHER"

    def _parse_args(self, argv):
        """
        Parse the command-line arguments into a dict.
        """

        opts = optparse.OptionParser()

        opts.add_option("--count", default="100", help="(%default)")
        opts.add_option("--offset", default="0", help="(%default)")
        opts.add_option(
            "--data-path", default="nyt-ingredients-snapshot-2015.csv", help="(%default)")

        (options, args) = opts.parse_args(argv)
        return options
