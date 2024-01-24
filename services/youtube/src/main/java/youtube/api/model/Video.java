package youtube.api.model;

import java.time.Duration;
import java.util.List;
import java.util.Objects;

/**
 * A video resource represents a YouTube video.
 */
public class Video implements Comparable<Video>{

    private String id;

    /**
     * The snippet object contains basic details about the video, such as its title, description, and category.
     */
    private SearchResultSnippet snippet;

    /**
     * The type of the video. Possible values for this property are:
     *
     * <ul>
     *     <li>episode</li>
     *     <li>movie</li>
     *     <li>other</li>
     * </ul>
     */
    private String type;

    /**
     * The category of this video. It contains the id and the title of the category. These properties are relevant
     * when using 'relatedToVideoId' filter or 'videoCategoryId' parameter in search list operation.
     */
    private VideoCategory category;

    /**
     * The language of the text in the video resource's snippet.title and snippet.description properties.
     */
    private String defaultLanguage;

    private Duration contentDetailsDuration;

    private VideoDimension contentDetailsDimension;

    private VideoDefinition contentDetailsDefinition;

    private Boolean contentDetailsCaption;

    private List<String> contentDetailsRegionRestrictionAllowed;

    private List<String> contentDetailsRegionRestrictionBlocked;

    private Boolean embeddable;

    private VideoLicense license;

    /**
     * A list of topic IDs that are relevant to the video.
     *
     * <br><br>
     *
     * Important: Due to the deprecation of Freebase and the Freebase API, topic IDs started working differently as of
     * February 27, 2017. At that time, YouTube started returning a small set of curated topic IDs.
     *
     * <br><br>
     *
     * <a href="https://developers.google.com/youtube/v3/docs/videos#topicDetails.relevantTopicIds[]" role="button">Link to documentation</a>
     */
    private List<String> relevantTopicIds;

    /**
     * @deprecated Deprecated as of June 1, 2017
     */
    @Deprecated
    private Double latitude;

    /**
     * @deprecated Deprecated as of June 1, 2017
     */
    @Deprecated
    private Double longitude;

    private Long viewCount;

    private Long likeCount;

    private Long dislikeCount;

    /**
     * A rating that YouTube uses to identify age-restricted content.
     * Valid values for this property are:
     *
     * <ul><li>ytAgeRestricted</li></ul>
     */
    private String ytRating;

    private Boolean hasLiveStreamingDetails;

    private Long liveStreamingConcurrentViewers;

    private Boolean forDeveloperVideo;

    public Video(String id, SearchResultSnippet snippet, String type, VideoCategory category, String defaultLanguage, Duration contentDetailsDuration, VideoDimension contentDetailsDimension,
                 VideoDefinition contentDetailsDefinition, Boolean contentDetailsCaption, List<String> contentDetailsRegionRestrictionAllowed, List<String> contentDetailsRegionRestrictionBlocked,
                 Boolean embeddable, VideoLicense license, List<String> relevantTopicIds, Long viewCount, Long likeCount, Long dislikeCount, String ytRating, Boolean liveStreamingDetails,
                 Long liveStreamingConcurrentViewers, Boolean forDeveloperVideo) {
        this.id = id;
        this.snippet = snippet;
        this.type = type;
        this.category = category;
        this.defaultLanguage = defaultLanguage;
        this.contentDetailsDuration = contentDetailsDuration;
        this.contentDetailsDimension = contentDetailsDimension;
        this.contentDetailsDefinition = contentDetailsDefinition;
        this.contentDetailsCaption = contentDetailsCaption;
        this.contentDetailsRegionRestrictionAllowed = contentDetailsRegionRestrictionAllowed;
        this.contentDetailsRegionRestrictionBlocked = contentDetailsRegionRestrictionBlocked;
        this.embeddable = embeddable;
        this.license = license;
        this.relevantTopicIds = relevantTopicIds;
        this.viewCount = viewCount;
        this.likeCount = likeCount;
        this.dislikeCount = dislikeCount;
        this.ytRating = ytRating;
        this.hasLiveStreamingDetails = liveStreamingDetails;
        this.liveStreamingConcurrentViewers = liveStreamingConcurrentViewers;
        this.forDeveloperVideo = forDeveloperVideo;
    }

    /**
     * @deprecated
     */
    @Deprecated
    public Video(String id, SearchResultSnippet snippet, String type, VideoCategory category, String defaultLanguage, Duration contentDetailsDuration, VideoDimension contentDetailsDimension,
                 VideoDefinition contentDetailsDefinition, Boolean contentDetailsCaption, List<String> contentDetailsRegionRestrictionAllowed, List<String> contentDetailsRegionRestrictionBlocked,
                 Boolean embeddable, VideoLicense license, List<String> relevantTopicIds, Double latitude, Double longitude, Long viewCount, Long likeCount, Long dislikeCount, String ytRating,
                 Boolean liveStreamingDetails, Long liveStreamingConcurrentViewers, Boolean forDeveloperVideo) {
        this(id, snippet, type, category, defaultLanguage, contentDetailsDuration, contentDetailsDimension, contentDetailsDefinition, contentDetailsCaption, contentDetailsRegionRestrictionAllowed,
                contentDetailsRegionRestrictionBlocked, embeddable, license, relevantTopicIds, viewCount, likeCount, dislikeCount, ytRating, liveStreamingDetails, liveStreamingConcurrentViewers,
                forDeveloperVideo);
        this.latitude = latitude;
        this.longitude = longitude;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public SearchResultSnippet getSnippet() {
        return snippet;
    }

    public void setSnippet(SearchResultSnippet snippet) {
        this.snippet = snippet;
    }

    public String getType() {
        return type;
    }

    /**
     * Set a type for the video. Possible values for this property are:
     *
     * <ul>
     *     <li>episode</li>
     *     <li>movie</li>
     *     <li>other</li>
     * </ul>
     */
    public void setType(String type) {
        this.type = type;
    }

    public VideoCategory getCategory() {
        return category;
    }

    public void setCategory(VideoCategory category) {
        this.category = category;
    }

    public String getDefaultLanguage() {
        return defaultLanguage;
    }

    public void setDefaultLanguage(String defaultLanguage) {
        this.defaultLanguage = defaultLanguage;
    }

    public Duration getContentDetailsDuration() {
        return contentDetailsDuration;
    }

    public VideoDimension getContentDetailsDimension() {
        return contentDetailsDimension;
    }

    public VideoDefinition getContentDetailsDefinition() {
        return contentDetailsDefinition;
    }

    public Boolean getContentDetailsCaption() {
        return contentDetailsCaption;
    }

    public List<String> getContentDetailsRegionRestrictionAllowed() {
        return contentDetailsRegionRestrictionAllowed;
    }

    public List<String> getContentDetailsRegionRestrictionBlocked() {
        return contentDetailsRegionRestrictionBlocked;
    }

    public Boolean getEmbeddable() {
        return embeddable;
    }

    public VideoLicense getLicense() {
        return license;
    }

    public List<String> getRelevantTopicIds() {
        return relevantTopicIds;
    }

    public Double getLatitude() {
        return latitude;
    }

    public Double getLongitude() {
        return longitude;
    }

    public Long getViewCount() {
        return viewCount;
    }

    public Long getLikeCount() {
        return likeCount;
    }

    public Long getDislikeCount() {
        return dislikeCount;
    }

    public String getYtRating() {
        return ytRating;
    }

    public Boolean getHasLiveStreamingDetails() {
        return hasLiveStreamingDetails;
    }

    public Long getLiveStreamingConcurrentViewers() {
        return liveStreamingConcurrentViewers;
    }

    public Boolean getForDeveloperVideo() {
        return forDeveloperVideo;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Video video = (Video) o;
        return id.equals(video.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public int compareTo(Video o) {
        int res = this.getViewCount().compareTo(o.getViewCount());
        if(res == 0) {
            res = this.getLikeCount().compareTo(o.getLikeCount());
            if(res == 0) {
                res = this.getDislikeCount().compareTo(o.getDislikeCount());
            }
        }
        return res;
    }

    public enum VideoDimension {
        TWO_D("2d"), THREE_D("3d");

        private String dimension;

        VideoDimension(String dimension) {
            this.dimension = dimension;
        }

        public String getDimension() {
            return dimension;
        }

        @Override
        public String toString() {
            return getDimension();
        }
    }

    public enum VideoDefinition {
        HD("high"), SD("standard");

        private String definition;

        VideoDefinition(String definition) {
            this.definition = definition;
        }

        public String getDefinition() {
            return definition;
        }

        @Override
        public String toString() {
            return getDefinition();
        }
    }

    public enum VideoLicense {
        YOUTUBE("youtube"), CREATIVE_COMMON("creativeCommon");

        private String license;

        VideoLicense(String license) {
            this.license = license;
        }

        public String getLicense() {
            return license;
        }

        @Override
        public String toString() {
            return license;
        }
    }
}


