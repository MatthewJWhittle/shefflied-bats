import ee


class Sentinel2CloudMasker:
    def __init__(
        self,
        aoi,
        start_date,
        end_date,
        cloud_filter=60,
        cld_prb_thresh=50,
        nir_drk_thresh=0.15,
        cld_prj_dist=1,
        buffer=50,
    ):
        self.aoi = aoi
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_filter = cloud_filter
        self.cld_prb_thresh = cld_prb_thresh
        self.nir_drk_thresh = nir_drk_thresh
        self.cld_prj_dist = cld_prj_dist
        self.buffer = buffer
        ee.Initialize()

    def get_s2_sr_cld_col(self):
        s2_sr_col = (
            ee.ImageCollection("COPERNICUS/S2_SR")
            .filterBounds(self.aoi)
            .filterDate(ee.Date(self.start_date), ee.Date(self.end_date))
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", self.cloud_filter))
        )
        s2_cloudless_col = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(self.aoi)
            .filterDate(ee.Date(self.start_date), ee.Date(self.end_date))
        )
        return ee.ImageCollection(
            ee.Join.saveFirst("s2cloudless").apply(
                primary=s2_sr_col,
                secondary=s2_cloudless_col,
                condition=ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            )
        )

    def add_cloud_bands(self, img):
        cld_prb = ee.Image(img.get("s2cloudless")).select("probability")
        is_cloud = cld_prb.gt(self.cld_prb_thresh).rename("clouds")
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def add_shadow_bands(self, img):
        # Identify water pixels from the SCL band.
        not_water = img.select("SCL").neq(6)

        # Identify dark NIR pixels that are not water (potential cloud shadow pixels).
        SR_BAND_SCALE = 1e4
        dark_pixels = (
            img.select("B8")
            .lt(self.nir_drk_thresh * SR_BAND_SCALE)
            .multiply(not_water)
            .rename("dark_pixels")
        )

        # Determine the direction to project cloud shadow from clouds (assumes UTM projection).
        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(img.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        # Project shadows from clouds for the distance specified by the CLD_PRJ_DIST input.
        cld_proj = (
            img.select("clouds")
            .directionalDistanceTransform(shadow_azimuth, self.cld_prj_dist * 10)
            .reproject(**{"crs": img.select(0).projection(), "scale": 100})
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        # Identify the intersection of dark pixels with cloud shadow projection.
        shadows = cld_proj.multiply(dark_pixels).rename("shadows")

        # Add dark pixels, cloud projection, and identified shadows as image bands.
        return img.addBands(ee.Image([dark_pixels, cld_proj, shadows]))

    def add_cld_shdw_mask(self, img):
        img_cloud = self.add_cloud_bands(img)
        img_cloud_shadow = self.add_shadow_bands(img_cloud)
        is_cld_shdw = (
            img_cloud_shadow.select("clouds")
            .add(img_cloud_shadow.select("shadows"))
            .gt(0)
        )
        is_cld_shdw = (
            is_cld_shdw.focalMin(2)
            .focalMax(self.buffer * 2 / 20)
            .reproject(crs=img.select([0]).projection(), scale=20)
            .rename("cloudmask")
        )
        return img_cloud_shadow.addBands(is_cld_shdw)

    def apply_cld_shdw_mask(self, img):
        not_cld_shdw = img.select("cloudmask").Not()
        return img.select("B.*").updateMask(not_cld_shdw)

    def get_cloud_masked_image(self):
        s2_sr_cld_col = self.get_s2_sr_cld_col()
        s2_sr_cld_col_masked = s2_sr_cld_col.map(self.add_cld_shdw_mask).map(
            self.apply_cld_shdw_mask
        )
        return s2_sr_cld_col_masked.median()


# Example usage:
# aoi = [-122.269, 45.701]
# start_date = '2020-06-01'
# end_date = '2020-06-02'
# masker = Sentinel2CloudMasker(aoi, start_date, end_date)
# cloud_masked_image = masker.get_cloud_masked_image()
