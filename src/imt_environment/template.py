class Template():
    def __init__(self, revised_hypo, tag):
        self.revised_hypo = revised_hypo
        self.tag = tag
        """ tag type
            0: delete
            1: keep
            2: insert
            3: replace
            4: blank
        """

    def template2hypo(self):
        # hypo = self.revised_hypo.strip().split()
        revised_hypo = self.revised_hypo[5:] if self.revised_hypo.startswith("<bos>") else self.revised_hypo
        assert len(revised_hypo) == len(self.tag), "\nrevised_hypo:{}\ntag: {}".format(revised_hypo, self.tag)
        hypo = [revised_hypo[i] for i in range(len(self.tag)) if 0 < self.tag[i] < 4]
        return "".join(hypo).lstrip(" ")

    def get_constraints(self):
        revised_hypo = self.revised_hypo[5:] if self.revised_hypo.startswith("<bos>") else self.revised_hypo
        constraints = []
        constraint = ""
        for i in range(len(self.tag)):
            if self.tag[i] == 0:
                continue
            elif self.tag[i] < 4:
                constraint += revised_hypo[i]
            elif constraint:
                constraints.append(constraint)
                constraint = ""
        if constraint and self.tag[i] < 4:
            constraints.append(constraint)
        return constraints