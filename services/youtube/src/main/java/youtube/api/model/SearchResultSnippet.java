package youtube.api.model;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Date;

@JsonPropertyOrder({"publishedAt", "channelId", "title", "description", "thumbnails", "channelTitle", "liveBroadcastContent"})
public class SearchResultSnippet {

    private String channelId;

    private String channelTitle;

    private String description;

    private LiveBroadcastContent liveBroadcastContent;

    private Date publishedAt;

    private ThumbnailDetails thumbnails;

    private String title;



    public String getChannelId() {
        return channelId;
    }

    public void setChannelId(String channelId) {
        this.channelId = channelId;
    }

    public String getChannelTitle() {
        return channelTitle;
    }

    public void setChannelTitle(String channelTitle) {
        this.channelTitle = channelTitle;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public LiveBroadcastContent getLiveBroadcastContent() {
        return liveBroadcastContent;
    }

    public void setLiveBroadcastContent(LiveBroadcastContent liveBroadcastContent) {
        this.liveBroadcastContent = liveBroadcastContent;
    }

    public Date getPublishedAt() {
        return publishedAt;
    }

    public void setPublishedAt(Date publishedAt) {
        this.publishedAt = publishedAt;
    }

    public ThumbnailDetails getThumbnails() {
        return thumbnails;
    }

    public void setThumbnails(ThumbnailDetails thumbnails) {
        this.thumbnails = thumbnails;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public enum LiveBroadcastContent {
        LIVE("live"), NONE("none"), UPCOMING("upcoming");

        private final String type;

        LiveBroadcastContent(String type) {
            this.type = type;
        }

        @JsonValue
        public String getType() {
            return type;
        }

        @Override
        public String toString() {
            return this.type;
        }
    }
}
