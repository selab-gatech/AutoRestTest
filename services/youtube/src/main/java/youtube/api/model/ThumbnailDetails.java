package youtube.api.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;


@JsonInclude(JsonInclude.Include.NON_NULL)
public class ThumbnailDetails {

    private Thumbnail defaultThumbnail;

    private Thumbnail standard;

    private Thumbnail medium;

    private Thumbnail high;

    private Thumbnail maxres;

    @JsonProperty("default")
    public Thumbnail getDefaultThumbnail() {
        return defaultThumbnail;
    }

    public void setDefaultThumbnail(Thumbnail defaultThumbnail) {
        this.defaultThumbnail = defaultThumbnail;
    }

    public Thumbnail getStandard() {
        return standard;
    }

    public void setStandard(Thumbnail standard) {
        this.standard = standard;
    }

    public Thumbnail getMedium() {
        return medium;
    }

    public void setMedium(Thumbnail medium) {
        this.medium = medium;
    }

    public Thumbnail getHigh() {
        return high;
    }

    public void setHigh(Thumbnail high) {
        this.high = high;
    }

    public Thumbnail getMaxres() {
        return maxres;
    }

    public void setMaxres(Thumbnail maxres) {
        this.maxres = maxres;
    }

}
