from collections import defaultdict, Counter
from itertools import chain


class GraphKeyFrame(object):
    def __init__(self):
        self.id = None
        self.meas = dict()
        self.covisible = defaultdict(int)

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphKeyFrame) and
                self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id  # predate

    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):

        return self.meas.keys()

    def mappoints(self):

        return self.meas.values()

    def add_measurement(self, m):

        self.meas[m] = m.mappoint

    def remove_measurement(self, m):

        try:
            del self.meas[m]
        except KeyError:
            pass

    def covisibility_keyframes(self):

        return self.covisible.copy()  # shallow copy

    def add_covisibility_keyframe(self, kf):

        self.covisible[kf] += 1


class GraphMapPoint(object):
    def __init__(self):
        self.id = None
        self.meas = dict()

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMapPoint) and
                self.id == rhs.id)

    def __lt__(self, rhs):
        return self.id < rhs.id

    def __le__(self, rhs):
        return self.id <= rhs.id

    def measurements(self):

        return self.meas.keys()

    def keyframes(self):

        return self.meas.values()

    def add_measurement(self, m):

        self.meas[m] = m.keyframe

    def remove_measurement(self, m):

        try:
            del self.meas[m]
        except KeyError:
            pass


class GraphMeasurement(object):
    def __init__(self):
        self.keyframe = None
        self.mappoint = None

    @property
    def id(self):
        return (self.keyframe.id, self.mappoint.id)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, rhs):
        return (isinstance(rhs, GraphMeasurement) and
                self.id == rhs.id)


class CovisibilityGraph(object):
    def __init__(self, ):
        self.kfs = []
        self.pts = set()

        self.kfs_set = set()
        self.meas_lookup = dict()

    def keyframes(self):
        return self.kfs.copy()

    def mappoints(self):
        return self.pts.copy()

    def add_keyframe(self, kf):
        self.kfs.append(kf)
        self.kfs_set.add(kf)

    def add_mappoint(self, pt):

        self.pts.add(pt)

    def remove_mappoint(self, pt):

        try:
            for m in pt.measurements():
                m.keyframe.remove_measurement(m)
                del self.meas_lookup[m.id]
            self.pts.remove(pt)
        except:
            pass

    def add_measurement(self, kf, pt, meas):
        if kf not in self.kfs_set or pt not in self.pts:
            return
        for m in pt.measurements():
            if m.keyframe == kf:
                continue
            kf.add_covisibility_keyframe(m.keyframe)
            m.keyframe.add_covisibility_keyframe(kf)

        meas.keyframe = kf
        meas.mappoint = pt
        kf.add_measurement(meas)
        pt.add_measurement(meas)

        self.meas_lookup[meas.id] = meas

    def remove_measurement(self, m):
        m.keyframe.remove_measurement(m)
        m.mappoint.remove_measurement(m)
        try:
            del self.meas_lookup[m.id]
        except:
            pass

    def has_measurement(self, *args):
        if len(args) == 1:  # measurement
            return args[0].id in self.meas_lookup
        elif len(args) == 2:  # keyframe, mappoint
            id = (args[0].id, args[1].id)
            return id in self.meas_lookup
        else:
            raise TypeError

    def get_reference_frame(self, seedpoints):
        assert len(seedpoints) > 0
        visible = [pt.keyframes() for pt in seedpoints]
        visible = Counter(chain(*visible))
        return visible.most_common(1)[0][0]

    def get_local_map_v2(self, seedframes, window_size=12, loop_window_size=8):
        # TODO: add similar (in appearance and location) keyframes' mappoints
        covisible = []
        for kf in set(seedframes):
            covisible.append(Counter(kf.covisibility_keyframes()))
        covisible = sum(covisible, Counter())
        for kf in set(seedframes):
            covisible[kf] = float('inf')
        local = sorted(
            covisible.items(), key=lambda _: _[1], reverse=True)

        id = max([_.id for _ in covisible])
        loop_frames = [_ for _ in local if _[0].id < id - 20]

        local = local[:window_size]
        loop_local = []
        if len(loop_frames) > 0:
            loop_covisible = sorted(
                loop_frames[0][0].covisibility_keyframes().items(),
                key=lambda _: _[1], reverse=True)

            for kf, n in loop_covisible:
                if kf not in set([_[0] for _ in local]):
                    loop_local.append((kf, n))
                    if len(loop_local) >= loop_window_size:
                        break

        local = chain(local, loop_local)

        local_map = []
        local_keyframes = []
        for kf, n in local:
            if n < 1:
                continue
            local_map.append(kf.mappoints())
            local_keyframes.append(kf)
        local_map = list(set(chain(*local_map)))
        return local_map, local_keyframes
