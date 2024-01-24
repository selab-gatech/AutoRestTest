package youtube.api.model;

import java.util.List;

public class Channel {

    private String id;

    /**
     * The snippet object contains basic details about the channel, such as its title, description, and category.
     */
    private SearchResultSnippet snippet;

    /**
     * The type of the channel. Possible values for this property are:
     *
     * <ul>
     *     <li>show</li>
     *     <li>other</li>
     * </ul>
     */
    private String type;

    private String defaultLanguage;

    private List<String> relevantTopicIds;

    private Long viewCount;

    private Long videoCount;

    private String contentOwnerId;

    public Channel(String id, SearchResultSnippet snippet, String type, String defaultLanguage, List<String> relevantTopicIds, Long viewCount, Long videoCount, String contentOwnerId) {
        this.id = id;
        this.snippet = snippet;
        this.type = type;
        this.defaultLanguage = defaultLanguage;
        this.relevantTopicIds = relevantTopicIds;
        this.viewCount = viewCount;
        this.videoCount = videoCount;
        this.contentOwnerId = contentOwnerId;
    }

    public String getId() {
        return id;
    }

    public SearchResultSnippet getSnippet() {
        return snippet;
    }

    public String getType() {
        return type;
    }

    public String getDefaultLanguage() {
        return defaultLanguage;
    }

    public List<String> getRelevantTopicIds() {
        return relevantTopicIds;
    }

    public Long getViewCount() {
        return viewCount;
    }

    public Long getVideoCount() {
        return videoCount;
    }

    public String getContentOwnerId() {
        return contentOwnerId;
    }
}
